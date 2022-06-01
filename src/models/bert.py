import tensorflow as tf
import tensorflow.keras as keras

from transformers import TFBertModel

from src.models.base import BaseModel


@BaseModel.register("bert")
class BertClassifier(BaseModel):
    def __init__(self, pretrained_bert_model, max_length, num_branchs):
        super().__init__()

        self.pretrained_bert_model = pretrained_bert_model
        self.max_length = max_length
        
        self.num_branchs = num_branchs
        
        self.graph = self.build_graph()

    def inputs(self):
        return [
            keras.layers.Input(shape=(self.max_length), dtype=tf.int32, name="input_ids"),
            keras.layers.Input(shape=(self.max_length), dtype=tf.int32, name="attention_mask")
        ]

    def build_graph(self):
        input_ids, attention_mask = self.inputs()

        bert_model = TFBertModel.from_pretrained(self.pretrained_bert_model)
        bert_output = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        last_hidden_state = bert_output[0]
        last_hidden_state = keras.layers.Dropout(0.1)(last_hidden_state)
        x_avg = keras.layers.GlobalAveragePooling1D()(last_hidden_state)
        x_max = keras.layers.GlobalMaxPooling1D()(last_hidden_state)
        x = keras.layers.Concatenate()([x_avg, x_max])

        branch_outputs = []
        for i in range(self.num_branchs):
            out = keras.layers.Dropout(0.5)(x)
            out = keras.layers.Dense(64, activation="relu",    name=f"branch_{i}/dense_1")(out)
            out = keras.layers.Dense(1,  activation="sigmoid", name=f"branch_{i}/dense_2")(out)
            branch_outputs.append(out)
        
        if self.num_branchs > 1:
            output = keras.layers.Average(name="output")(branch_outputs)
        else:
            output = branch_outputs[0]

        model = keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
        return model
