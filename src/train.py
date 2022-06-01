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
from transformers import BertTokenizer
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


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

    seed = config["seed"]

    tokenizer = BertTokenizer.from_pretrained(model_config["pretrained_bert_model"], do_lower_case=True)

    max_length = model_config["max_length"]
    train_df = pd.read_csv(data_config["path"]["train"], sep="\t")

    train_input_ids, train_attention_masks = encode_aug(tokenizer, train_df, max_length)

    dataset = [[ids, att] for ids, att in zip(train_input_ids, train_attention_masks)]
    labels = train_df.target.tolist()

    train_data, val_data, train_labels, val_labels = train_test_split(
        dataset, labels, train_size=0.9, random_state=seed, stratify=labels)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (collections.OrderedDict({"input_ids": [x[0] for x in train_data],
                                 "attention_mask": [x[1] for x in train_data]}), train_labels))\
            .shuffle(buffer_size=trainer_config["batch_size"] * 20, seed=seed)\
            .batch(trainer_config["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (collections.OrderedDict({"input_ids": [x[0] for x in val_data],
                                 "attention_mask": [x[1] for x in val_data]}), val_labels))\
            .shuffle(buffer_size=trainer_config["batch_size"] * 20, seed=seed)\
            .batch(trainer_config["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', min_delta=0.001, patience=5, verbose=1, mode='max',
        baseline=None, restore_best_weights=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weight_dir, "best_weights"), save_weights_only=True,
        monitor='val_binary_accuracy', mode='max', save_best_only=True)

    model = BaseModel.from_params(model_config)
    if recover:
        model.graph.load_weights(weight_dir)

    model.graph.compile(
        optimizer=get_optimizer(trainer_config["optimizer"]),
        loss=build_loss_fn(trainer_config),
        metrics=tf.keras.metrics.BinaryAccuracy())

    model.graph.fit(
        train_dataset, validation_data=val_dataset, epochs=trainer_config["num_epochs"],
        callbacks=[early_stopping_callback, model_checkpoint_callback])


def eval(checkpoint_path, dataset_path):
    raise NotImplementedError()


def create_submission(checkpoint_dir):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]

    tokenizer = BertTokenizer.from_pretrained(model_config["pretrained_bert_model"], do_lower_case=True)

    test_df = pd.read_csv(data_config["path"]["test"], sep="\t")
    test_input_ids, test_attention_masks = encode_aug(tokenizer, test_df, model_config["max_length"])
    ids = test_df.id.tolist()

    model = BaseModel.from_params(model_config)
    model.graph.load_weights(os.path.join(checkpoint_dir, "checkpoints/best_weights"))

    submission_file = os.path.join(checkpoint_dir, "submission.csv")
    predictions = {"id": [], "target": []}
    
    input_ids_batch = []
    attn_mask_batch = []
    batch_size = 256
    for i in tqdm(range(len(ids))):
        predictions["id"].append(ids[i])
        input_ids_batch.append(test_input_ids[i])
        attn_mask_batch.append(test_attention_masks[i])
        if len(input_ids_batch) == batch_size:
            input_ids_batch, attn_mask_batch = np.array(input_ids_batch), np.array(attn_mask_batch)
            prob = model.graph([input_ids_batch, attn_mask_batch]).numpy()
            predictions["target"].extend([round(prob[j][0]) for j in range(batch_size)])
            
            input_ids_batch = []
            attn_mask_batch = []

    remain = len(input_ids_batch)
    if remain:
        input_ids_batch, attn_mask_batch = np.array(input_ids_batch), np.array(attn_mask_batch)
        prob = model.graph([input_ids_batch, attn_mask_batch]).numpy()
        predictions["target"].extend([round(prob[j][0]) for j in range(remain)])

    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(submission_file, index=False)


def hyperparams_search(config_file, force=False):
    config_file_name = os.path.splitext(os.path.split(config_file)[1])[0]
    config = Params.from_file(config_file)
    hyp = config["hyp"]

    num_hyp = len(hyp)
    inds = [0] * num_hyp
    keys_list = []
    values_list = []
    len_search = []
    for k, v in hyp.items():
        keys_list.append(k)
        values_list.append(v)
        len_search.append(len(v))

    best_metric = 0
    best_model_dir = ""
    saved_model_list = []
    i = 0
    name = []
    while True:
        if inds[i] == len_search[i]:
            if i == 0:
                break
            inds[i] = 0
            i -= 1
            name.pop()
            continue

        if i == 0:
            name = []
        main_key, sub_key = keys_list[i].split(".")
        values = values_list[i][inds[i]]
        config[main_key][sub_key] = values
        inds[i] += 1
        name.append(sub_key + "-" + str(values))
        if i == num_hyp - 1:
            print(config.__dict__["params"])
            config_path = os.path.join("/tmp", config_file_name + '_' + '_'.join(name) + ".json")
            name.pop()
            save_json(config_path, config.__dict__["params"])

            model_dir = train()
            saved_model_list.append(model_dir)

            metric = eval()
            save_txt(os.path.join(model_dir, "res.txt"), [metric])
            if metric > best_metric:
                best_metric = metric
                best_model_dir = model_dir
        else:
            i += 1

    print("============================================")
    print("Best results: ", best_metric)
    print(best_model_dir + "\n\n")

    for d in saved_model_list:
        if d == best_model_dir:
            continue
        shutil.rmtree(d)
