from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys

import numpy as np
import tensorflow as tf
from fastprogress import master_bar, progress_bar
from seqeval.metrics import classification_report

from model import BertNer
from optimization import AdamWeightDecay, WarmUp
from tokenization import FullTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(True)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, True)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(True)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [True] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(False)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(False)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-cased,bert-large-cased")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--eval_on",
                        default="dev",
                        type=str,
                        help="Evaluation set, dev: Development, test: Test")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    # training stratergy arguments
    parser.add_argument("--multi_gpu",
                        action='store_true',
                        help="Set this flag to enable multi-gpu training using MirroredStrategy."
                             "Single gpu training")
    parser.add_argument("--gpus",default='0',type=str,
                        help="Comma separated list of gpus devices."
                              "For Single gpu pass the gpu id.Default '0' GPU"
                              "For Multi gpu,if gpus not specified all the available gpus will be used")

    args = parser.parse_args()

    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.do_train:
        tokenizer = FullTokenizer(os.path.join(args.bert_model, "vocab.txt"), args.do_lower_case)
    
    if args.multi_gpu:
        if len(args.gpus.split(',')) == 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            gpus = [f"/gpu:{gpu}" for gpu in args.gpus.split(',')]
            strategy = tf.distribute.MirroredStrategy(devices=gpus)
    else:
        gpu = args.gpus.split(',')[0]
        strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{gpu}")

    train_examples = None
    optimizer = None
    num_train_optimization_steps = 0
    ner = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size) * args.num_train_epochs
        warmup_steps = int(args.warmup_proportion *
                           num_train_optimization_steps)
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.learning_rate,
                                                decay_steps=num_train_optimization_steps,end_learning_rate=0.0)
        if warmup_steps:
            learning_rate_fn = WarmUp(initial_learning_rate=args.learning_rate,
                                    decay_schedule_fn=learning_rate_fn,
                                    warmup_steps=warmup_steps)
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=args.weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=args.adam_epsilon,
            exclude_from_weight_decay=['layer_norm', 'bias'])

        with strategy.scope():
            ner = BertNer(args.bert_model, tf.float32, num_labels, args.max_seq_length)
            loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    label_map = {i: label for i, label in enumerate(label_list, 1)}
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_ids for f in train_features]))
        all_input_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_mask for f in train_features]))
        all_segment_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.segment_ids for f in train_features]))
        all_valid_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.valid_ids for f in train_features]))
        all_label_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_mask for f in train_features]))

        all_label_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_id for f in train_features]))

        # Dataset using tf.data
        train_data = tf.data.Dataset.zip(
            (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids,all_label_mask))
        shuffled_train_data = train_data.shuffle(buffer_size=int(len(train_features) * 0.1),
                                                seed = args.seed,
                                                reshuffle_each_iteration=True)
        batched_train_data = shuffled_train_data.batch(args.train_batch_size)
        # Distributed dataset
        dist_dataset = strategy.experimental_distribute_dataset(batched_train_data)

        loss_metric = tf.keras.metrics.Mean()

        epoch_bar = master_bar(range(args.num_train_epochs))
        pb_max_len = math.ceil(
            float(len(train_features))/float(args.train_batch_size))

        def train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask):
            def step_fn(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask):

                with tf.GradientTape() as tape:
                    logits = ner(input_ids, input_mask,segment_ids, valid_ids, training=True)
                    label_mask = tf.reshape(label_mask,(-1,))
                    logits = tf.reshape(logits,(-1,num_labels))
                    logits_masked = tf.boolean_mask(logits,label_mask)
                    label_ids = tf.reshape(label_ids,(-1,))
                    label_ids_masked = tf.boolean_mask(label_ids,label_mask)
                    cross_entropy = loss_fct(label_ids_masked, logits_masked)
                    loss = tf.reduce_sum(cross_entropy) * (1.0 / args.train_batch_size)
                grads = tape.gradient(loss, ner.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, ner.trainable_variables)))
                return cross_entropy

            per_example_losses = strategy.experimental_run_v2(step_fn,
                                     args=(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask))
            mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            return mean_loss

        for epoch in epoch_bar:
            with strategy.scope():
                for (input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask) in progress_bar(dist_dataset, total=pb_max_len, parent=epoch_bar):
                    loss = train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask)
                    loss_metric(loss)
                    epoch_bar.child.comment = f'loss : {loss_metric.result()}'
            loss_metric.reset_states()
        
        # model weight save 
        ner.save_weights(os.path.join(args.output_dir,"model.h5"))
        # copy vocab to output_dir
        shutil.copyfile(os.path.join(args.bert_model,"vocab.txt"),os.path.join(args.output_dir,"vocab.txt"))
        # copy bert config to output_dir
        shutil.copyfile(os.path.join(args.bert_model,"bert_config.json"),os.path.join(args.output_dir,"bert_config.json"))
        # save label_map and max_seq_length of trained model
        model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,
                        "max_seq_length":args.max_seq_length,"num_labels":num_labels,
                        "label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"),indent=4)
        

    if args.do_eval:
        # load tokenizer
        tokenizer = FullTokenizer(os.path.join(args.output_dir, "vocab.txt"), args.do_lower_case)
        # model build hack : fix
        config = json.load(open(os.path.join(args.output_dir,"bert_config.json")))
        ner = BertNer(config, tf.float32, num_labels, args.max_seq_length)
        ids = tf.ones((1,128),dtype=tf.int64)
        _ = ner(ids,ids,ids,ids, training=False)
        ner.load_weights(os.path.join(args.output_dir,"model.h5"))

        # load test or development set based on argsK
        if args.eval_on == "dev":
            eval_examples = processor.get_dev_examples(args.data_dir)
        elif args.eval_on == "test":
            eval_examples = processor.get_test_examples(args.data_dir)
        
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evalution *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_ids for f in eval_features]))
        all_input_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_mask for f in eval_features]))
        all_segment_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.segment_ids for f in eval_features]))
        all_valid_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.valid_ids for f in eval_features]))

        all_label_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_id for f in eval_features]))

        eval_data = tf.data.Dataset.zip(
            (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids))
        batched_eval_data = eval_data.batch(args.train_batch_size)

        loss_metric = tf.keras.metrics.Mean()
        epoch_bar = master_bar(range(1))
        pb_max_len = math.ceil(
            float(len(eval_features))/float(args.eval_batch_size))

        y_true = []
        y_pred = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        for epoch in epoch_bar:
            for (input_ids, input_mask, segment_ids, valid_ids, label_ids) in progress_bar(batched_eval_data, total=pb_max_len, parent=epoch_bar):
                    logits = ner(input_ids, input_mask,
                                 segment_ids, valid_ids, training=False)
                    logits = tf.argmax(logits,axis=2)
                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j,m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == len(label_map):
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j].numpy()])
                                temp_2.append(label_map.get(logits[i][j].numpy(),'O'))
            report = classification_report(y_true, y_pred,digits=4)       
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                logger.info("\n%s", report)
                writer.write(report)

if __name__ == "__main__":
    main()
