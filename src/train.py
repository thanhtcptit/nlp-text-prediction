import os
import time
import shutil
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from src.utils.train_utils import *
from src.models.base import BaseModel
from src.utils import Params, save_json, save_txt, load_json

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

    callbacks = []
    if "callbacks" in trainer_config:
        for callback in trainer_config["callbacks"]:
            if "params" not in callback:
                callback["params"] = {}
            if callback["type"] == "model_checkpoint":
                callback["params"]["filepath"] = os.path.join(weight_dir, "best.ckpt")
            elif callback["type"] == "logging":
                callback["params"]["filename"] = os.path.join(checkpoint_dir, "log.csv")
            callbacks.append(get_callback_fn(callback["type"])(**callback["params"]))

    model = BaseModel.from_params(model_config).build_graph()
    if recover:
        model.load_weights(weight_dir)

    model.compile(
        optimizer=get_optimizer(trainer_config["optimizer"]["type"])(**trainer_config["optimizer"].get("params", {})),
        loss=get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {})),
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
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))
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

        callbacks = []
        if "callbacks" in trainer_config:
            for callback in trainer_config["callbacks"]:
                callbacks.append(get_callback_fn(callback["type"])(**callback["params"]))
        callbacks.append(TFKerasPruningCallback(trial, "val_binary_accuracy"))

        model = BaseModel.from_params(model_config).build_graph_for_hp(trial)

        optimizer = trial.suggest_categorical("optimizer", trainer_config["optimizer"]["type"])
        lr = trial.suggest_float("lr", trainer_config["optimizer"]["params"]["learning_rate"][0],
                                 trainer_config["optimizer"]["params"]["learning_rate"][1], log=True)
        model.compile(
            optimizer=get_optimizer(optimizer)(learning_rate=lr),
            loss=get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {})),
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

    df = study.trials_dataframe()
    print(df)

    print("Best trial:")
    trial = study.best_trial

    print(" - Value: ", trial.value)
    print(" - Params: ")
    for key, value in trial.params.items():
        print("  - {}: {}".format(key, value))
