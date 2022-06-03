import os
import time
import shutil
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
from src.models.base import BaseModel
import tensorflow.keras as keras

from tqdm import tqdm
from transformers import AutoTokenizer
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score

from src.utils.train_utils import *
from src.utils import Params, save_json, save_txt, load_json, Logger

tf.get_logger().setLevel('INFO')


def encode_aug(tokenizer, data, max_length):
    # TODO Implement random sentence augment
    input_ids = []
    attention_masks = []
  
    for i in range(len(data.text)):
        encoded = tokenizer.encode_plus(data.text[i], add_special_tokens=True,
                                        max_length=max_length, pad_to_max_length=True)
        token_len = sum(encoded["attention_mask"])
        if token_len > max_length * 0.8:
            encoded = tokenizer.encode_plus(data.text[i], add_special_tokens=True)
            encoded_ids = encoded["input_ids"]
            encoded_attention = encoded["attention_mask"]  
            max_len_half = int(max_length / 2)
            input_ids.append(encoded_ids[:max_len_half] + encoded_ids[-max_len_half:])
            attention_masks.append(encoded_attention[:max_len_half] + encoded_attention[-max_len_half:])
        else:  
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
    return input_ids, attention_masks


def focal_loss(gamma=2.0, alpha=0.2):
    def focal_loss_fn(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) \
            * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fn


def build_loss_fn(config):
    if config["loss_fn"] == "bce":
        return keras.losses.BinaryCrossentropy(label_smoothing=config.get("label_smoothing", 0))
    if config["loss_fn"] == "cce":
        return keras.losses.CategoricalCrossentropy(label_smoothing=config.get("label_smoothing", 0))
    if config["loss_fn"] == "mse":
        return keras.losses.MeanSquaredError()
    if config["loss_fn"] == "focal":
        return focal_loss(config.get("gamma", 2.0), config.get("alpha", 0.2))
    raise ValueError(f"Can't identify loss_fn {config['loss_fn']}")


def create_tf_dataset(df, tokenizer, max_length, batch_size, is_train=False):
    input_ids, attention_masks = encode_aug(tokenizer, df, max_length)
    labels = df.target.tolist()
    
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        (collections.OrderedDict({"input_ids": input_ids, "attention_mask": attention_masks}),
         labels))
    if is_train:
        tf_dataset = tf_dataset.shuffle(buffer_size=batch_size * 100)
    tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset


def train(config_path, checkpoint_dir, recover=False, force=False):
    if not checkpoint_dir:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        checkpoint_dir = os.path.join("train_logs", config_name)
    if os.path.exists(checkpoint_dir):
        if force:
            shutil.rmtree(checkpoint_dir)
    weight_dir = os.path.join(checkpoint_dir, "checkpoints")
    os.makedirs(weight_dir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(checkpoint_dir, "config.json"))

    config = Params.from_file(config_path)
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_bert_model"])

    train_df = pd.read_csv(data_config["path"]["train"], sep="\t")
    val_df = pd.read_csv(data_config["path"]["val"], sep="\t")

    train_dataset = create_tf_dataset(train_df, tokenizer, model_config["max_length"],
                                      trainer_config["batch_size"], is_train=True)
    val_dataset = create_tf_dataset(val_df, tokenizer, model_config["max_length"],
                                    trainer_config["batch_size"])

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', min_delta=0.001, patience=5, verbose=1, mode='max',
        baseline=None, restore_best_weights=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weight_dir, "best_weights"), save_weights_only=True,
        monitor='val_binary_accuracy', mode='max', save_best_only=True)

    logging_callback = tf.keras.callbacks.CSVLogger(os.path.join(checkpoint_dir, "logs.csv"),
                                                    separator='\t', append=True)

    callbacks = [early_stopping_callback, model_checkpoint_callback, logging_callback]
    if "callbacks" in trainer_config:
        if "lr_scheduler" in trainer_config["callbacks"]:
            callbacks.append(get_lr_scheduler(trainer_config["callbacks"]["lr_scheduler"]))

    model = BaseModel.from_params(model_config).build_graph()
    if recover:
        model.load_weights(weight_dir)

    model.compile(
        optimizer=get_optimizer(trainer_config["optimizer"])(
            learning_rate=trainer_config["optimizer"]["learning_rate"]),
        loss=build_loss_fn(trainer_config),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])

    model.fit(
        train_dataset, validation_data=val_dataset, epochs=trainer_config["num_epochs"],
        callbacks=callbacks)

    metrics = model.evaluate(val_dataset)
    print(metrics)

    create_submission(checkpoint_dir, model, threshold=None)
    return metrics


def eval(checkpoint_dir, dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_bert_model"], do_lower_case=True)

    if not dataset_path:
        dataset_path = data_config["path"]["val"]
    test_df = pd.read_csv(dataset_path, sep="\t")
    test_dataset = create_tf_dataset(test_df, tokenizer, model_config["max_length"],
                                     trainer_config["batch_size"])

    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best_weights"))
    model.compile(
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
    metrics = model.evaluate(test_dataset)
    print(metrics)
    return metrics


def create_submission(checkpoint_dir, model=None, threshold=0.5):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]

    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_bert_model"], do_lower_case=True)
    max_length = model_config["max_length"]
    if not model:
        model = BaseModel.from_params(model_config).build_graph()
        model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best_weights"))

    def predict(df):
        input_ids, attention_masks = encode_aug(tokenizer, df, max_length)
        probs, input_ids_batch, attn_mask_batch = [], [], []
        batch_size = 256

        for i in tqdm(range(len(input_ids))):
            input_ids_batch.append(input_ids[i])
            attn_mask_batch.append(attention_masks[i])
            if len(input_ids_batch) == batch_size:
                input_ids_batch, attn_mask_batch = np.array(input_ids_batch), np.array(attn_mask_batch)
                prob = model([input_ids_batch, attn_mask_batch]).numpy()
                probs.extend([prob[j][0] for j in range(batch_size)])

                input_ids_batch = []
                attn_mask_batch = []

        remain = len(input_ids_batch)
        if remain:
            input_ids_batch, attn_mask_batch = np.array(input_ids_batch), np.array(attn_mask_batch)
            prob = model([input_ids_batch, attn_mask_batch]).numpy()
            probs.extend([prob[j][0] for j in range(remain)])
        return probs


    if threshold is None:
        thresholds = np.arange(0.2, 0.9, step=0.05)

        val_df = pd.read_csv(data_config["path"]["val"], sep="\t")
        y_true = val_df.target.tolist()
        probs = predict(val_df)

        best_val = 0
        for t in thresholds:
            y_pred = [1 if v >= t else 0 for v in probs]
            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_val:
                best_val = f1
                threshold = t
        print(f"Best f1: {best_val} at threshold: {threshold}")

    test_df = pd.read_csv(data_config["path"]["test"], sep="\t")
    test_ids = test_df.id.tolist()
    probs = predict(test_df)

    submission_file = os.path.join(checkpoint_dir, "submission.csv")
    predictions = {"id": test_ids, "target": [1 if v >= threshold else 0 for v in probs]}
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(submission_file, index=False)


def hyperparams_search(config_file, num_trials=50, force=False):
    import optuna
    from optuna.integration import TFKerasPruningCallback

    def objective(trial):
        tf.keras.backend.clear_session()

        config = Params.from_file(config_file)
        data_config = config["data"]
        model_config = config["model"]
        trainer_config = config["trainer"]

        tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_bert_model"])

        train_df = pd.read_csv(data_config["path"]["train"], sep="\t")
        val_df = pd.read_csv(data_config["path"]["val"], sep="\t")

        train_dataset = create_tf_dataset(train_df, tokenizer, model_config["max_length"],
                                        trainer_config["batch_size"], is_train=True)
        val_dataset = create_tf_dataset(val_df, tokenizer, model_config["max_length"],
                                        trainer_config["batch_size"])

        monitor_val = "val_binary_accuracy"
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=monitor_val, min_delta=0.001, patience=5, verbose=1, mode='max',
            baseline=None, restore_best_weights=True)

        callbacks = [early_stopping_callback, TFKerasPruningCallback(trial, monitor_val)]
        if "callbacks" in trainer_config:
            if "lr_scheduler" in trainer_config["callbacks"]:
                callbacks.append(get_lr_scheduler(trainer_config["callbacks"]["lr_scheduler"]))

        model = BaseModel.from_params(model_config).build_graph_for_hp(trial)
        lr = trial.suggest_float("lr", trainer_config["optimizer"]["learning_rate"][0],
                                 trainer_config["optimizer"]["learning_rate"][1], log=True)
        model.compile(
            optimizer=get_optimizer(trainer_config["optimizer"])(learning_rate=lr),
            loss=build_loss_fn(trainer_config),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
        model.summary()

        model.fit(
            train_dataset, validation_data=val_dataset, epochs=trainer_config["num_epochs"],
            callbacks=callbacks)

        metrics = model.evaluate(val_dataset)
        return metrics[1]

    study = optuna.create_study(study_name="bert", direction="maximize")
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True,
                   catch=(tf.errors.InvalidArgumentError,))
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(" - Value: ", trial.value)
    print(" - Params: ")
    for key, value in trial.params.items():
        print("  - {}: {}".format(key, value))
