"""BERT NER Inference."""

from __future__ import absolute_import, division, print_function

import json
import os

import tensorflow as tf
from nltk import word_tokenize

from model import BertNer
from tokenization import FullTokenizer


class Ner:

    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        bert_config = json.load(open(os.path.join(model_dir,"bert_config.json")))
        model = BertNer(bert_config, tf.float32, model_config['num_labels'], model_config['max_seq_length'])
        ids = tf.ones((1,128),dtype=tf.int32)
        _ = model(ids,ids,ids,ids, training=False)
        model.load_weights(os.path.join(model_dir,"model.h5"))
        voacb = os.path.join(model_dir, "vocab.txt")
        tokenizer = FullTokenizer(vocab_file=voacb, do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        start_position = 1
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            valid_positions.append(start_position)
            start_position += len(token)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,0)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(valid_positions[-1]+1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # while len(input_ids) < self.max_seq_length:
        #     input_ids.append(0)
        #     input_mask.append(0)
        #     segment_ids.append(0)
        #     valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
        input_ids = tf.Variable([input_ids],dtype=tf.int32)
        input_mask = tf.Variable([input_mask],dtype=tf.int32)
        segment_ids = tf.Variable([segment_ids],dtype=tf.int32)
        valid_ids = tf.Variable([valid_ids],dtype=tf.int32)
        logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits_label = tf.argmax(logits,axis=2)
        logits_label = logits_label.numpy().tolist()[0]

        logits_confidence = [values[label].numpy() for values,label in zip(logits[0],logits_label)]

        logits = zip(logits_label[1:-1],logits_confidence[1:-1])

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
        return output
