import tensorflow as tf
import tensorflow.keras as keras

from transformers import TFAutoModel

from src.models.base import BaseModel


@BaseModel.register("bert")
class BertClassifier(BaseModel):
    def __init__(self, pretrained_bert_model, max_length, depth,
                 hidden_units, act, num_dropout_sample, drop_out, freeze_bert=False):
        super().__init__()

        self.pretrained_bert_model = pretrained_bert_model
        self.max_length = max_length

        self.depth = depth
        self.hidden_units = hidden_units
        self.act = act
        self.drop_out = drop_out
        self.num_dropout_sample = num_dropout_sample
        self.freeze_bert = freeze_bert

    def inputs(self):
        return [
            keras.layers.Input(shape=(self.max_length), dtype=tf.int32, name="input_ids"),
            keras.layers.Input(shape=(self.max_length), dtype=tf.int32, name="attention_mask")
        ]

    def build_graph(self):
        input_ids, attention_mask = self.inputs()

        bert_model = TFAutoModel.from_pretrained(self.pretrained_bert_model)
        if self.freeze_bert:
            for w in bert_model.weights:
                w._trainable = False

        bert_output = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        last_hidden_state = bert_output[0]
        last_hidden_state = keras.layers.Dropout(self.drop_out)(last_hidden_state)
        x_avg = keras.layers.GlobalAveragePooling1D()(last_hidden_state)
        x_max = keras.layers.GlobalMaxPooling1D()(last_hidden_state)
        x = keras.layers.Concatenate()([x_avg, x_max])

        outputs = []
        for i in range(self.num_dropout_sample):
            out = x
            for j in range(self.depth):
                out = keras.layers.Dense(self.hidden_units, activation=self.act,
                                         name=f"branch_{i}/dense_{j}")(out)
                out = keras.layers.Dropout(self.drop_out)(out)
            out = keras.layers.Dense(1, name=f"branch_{i}/fc")(out)
            outputs.append(out)

        if self.num_dropout_sample > 1:
            output = tf.math.sigmoid(keras.layers.Average(name="output")(outputs))
        else:
            output = tf.math.sigmoid(outputs[0])

        model = keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
        return model

    def build_graph_for_hp(self, trial):
        lhs_dropout = trial.suggest_float("lhs_dropout", self.drop_out[0], self.drop_out[1])
        num_dropout_sample = trial.suggest_int("num_dropout_sample", self.num_dropout_sample[0],
                                               self.num_dropout_sample[1])
        depth = trial.suggest_int(f"depth", self.depth[0], self.depth[1])
        hidden_units = trial.suggest_int(
            f"hidden_units", self.hidden_units[0], self.hidden_units[1])
        dense_dropout = trial.suggest_float(
            f"dense_dropout", self.drop_out[0], self.drop_out[1])
        act = trial.suggest_categorical("act", self.act)

        input_ids, attention_mask = self.inputs()

        bert_model = TFAutoModel.from_pretrained(self.pretrained_bert_model)
        if self.freeze_bert:
            for w in bert_model.weights:
                w._trainable = False

        bert_output = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        last_hidden_state = bert_output[0]
        last_hidden_state = keras.layers.Dropout(lhs_dropout)(last_hidden_state)
        x_avg = keras.layers.GlobalAveragePooling1D()(last_hidden_state)
        x_max = keras.layers.GlobalMaxPooling1D()(last_hidden_state)
        x = keras.layers.Concatenate()([x_avg, x_max])

        outputs = []
        for i in range(num_dropout_sample):
            out = x
            for j in range(depth):
                out = keras.layers.Dense(hidden_units, activation=act,
                                         name=f"branch_{i}/dense_{j}")(out)
                out = keras.layers.Dropout(dense_dropout)(out)
            out = keras.layers.Dense(1, name=f"branch_{i}/fc")(out)
            outputs.append(out)

        if len(outputs) > 1:
            output = tf.math.sigmoid(keras.layers.Average(name="output")(outputs))
        else:
            output = tf.math.sigmoid(outputs[0])

        model = keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
        return model
