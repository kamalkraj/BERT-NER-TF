from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

import bert_modeling as bert_model
from utils import tf_utils


class BertNer(tf.keras.Model):

    def __init__(self, bert_path, float_type, num_labels, max_seq_length, final_layer_initializer=None):
        super(BertNer, self).__init__()
        bert_config = bert_model.BertConfig.from_json_file(os.path.join(bert_path,"bert_config.json"))
        self.bert = bert_model.BertModel(config=bert_config,float_type=float_type)
        init_checkpoint = os.path.join(bert_path,"bert_model.ckpt")
        checkpoint = tf.train.Checkpoint(model=self.bert)
        checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
        if final_layer_initializer is not None:
            initializer = final_layer_initializer
        else:
            initializer = tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range)
        self.dropout = tf.keras.layers.Dropout(
            rate=bert_config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            num_labels, kernel_initializer=initializer, activation='softmax',name='output', dtype=float_type)
    
    # def __call__(self,
    #            input_word_ids,
    #            input_mask=None,
    #            input_type_ids=None,
    #            valid_mask=None,
    #            **kwargs):
    #     inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids, valid_mask])
    #     return super(BertNer, self).__call__(inputs, **kwargs)
    
    def call(self, inputs, **kwargs):
        # unpacked_inputs = tf_utils.unpack_inputs(inputs)
        input_word_ids = inputs[0]
        input_mask = inputs[1]
        input_type_ids = inputs[2]
        valid_mask = inputs[3]
        _, sequence_output = self.bert(input_word_ids, input_mask, input_type_ids)
        valid_output = []
        for i in range(sequence_output.shape[0]):
            r = 0
            for j in range(sequence_output.shape[1]):
                if valid_mask[i][j] == 1:
                    valid_output = valid_output + [sequence_output[i][j]]
                else:
                    r += 1
            for _ in tf.range(r):
                valid_output = valid_output + [tf.zeros_like(sequence_output[i][j])]
        valid_output = tf.reshape(tf.stack(valid_output),sequence_output.shape)
        sequence_output = self.dropout(
            valid_output, training=kwargs.get('training', False))
        logits = self.classifier(sequence_output)
        return logits
