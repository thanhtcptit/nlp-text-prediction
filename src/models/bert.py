import tensorflow as tf
import tensorflow.keras as keras

from transformers import TFAutoModel

from src.models.base import BaseModel


@BaseModel.register("bert")
class BertClassifier(BaseModel):
    def __init__(self, pretrained_bert_model, max_length, num_dropout_sample,
                 hidden_units=64, freeze_bert=False):
        super().__init__()

        self.pretrained_bert_model = pretrained_bert_model
        self.max_length = max_length

        self.num_dropout_sample = num_dropout_sample
        self.hidden_units = hidden_units
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
        last_hidden_state = keras.layers.Dropout(0.1)(last_hidden_state)
        x_avg = keras.layers.GlobalAveragePooling1D()(last_hidden_state)
        x_max = keras.layers.GlobalMaxPooling1D()(last_hidden_state)
        x = keras.layers.Concatenate()([x_avg, x_max])

        outputs = []
        for i in range(self.num_dropout_sample):
            out = keras.layers.Dropout(0.5)(x)
            out = keras.layers.Dense(self.hidden_units, activation="relu", name=f"branch_{i}/dense_1")(out)
            out = keras.layers.Dense(1, activation="sigmoid", name=f"branch_{i}/dense_2")(out)
            outputs.append(out)

        if self.num_dropout_sample > 1:
            output = keras.layers.Average(name="output")(outputs)
        else:
            output = outputs[0]

        model = keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
        return model
