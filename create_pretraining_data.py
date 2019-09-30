# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM TF examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import random
import tensorflow as tf
from bert import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, mention_id, word_ids):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.mention_id = mention_id
    self.word_ids = word_ids

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "mention_id: %s\n" % (" ".join([str(x) for x in self.mention_id]))
    s += "word_ids: %s\n" % (" ".join([str(x) for x in self.word_ids]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    mention_id = list(instance.mention_id)
    word_ids = list(instance.word_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      mention_id.append(0)
      word_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(mention_id) == max_seq_length
    assert len(word_ids) == max_seq_length

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["mention_id"] = create_int_feature(mention_id)
    features["word_ids"] = create_int_feature(word_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])

        line = json.loads(line)['text']
        tokens = tokenizer.tokenize(line)

        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for document_index in range(len(all_documents)):
    instances.extend(
        create_instances_from_document(
            all_documents, document_index, max_seq_length, vocab_words, rng))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  mention_length = int(max_num_tokens/2)

  instances = []
  for doc in document:
    for i in range(int(len(doc)/(0.5*max_num_tokens))):

      start_pos = int(0.5*max_num_tokens*i)
      tokens = doc[start_pos: start_pos+max_num_tokens]

      tokens_a = tokens[:mention_length]
      tokens_b = tokens[mention_length:]

      if not tokens_a or not tokens_b:
        continue

      tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
      segment_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b) + 1)
      mention_id = [0]*len(tokens)
      word_ids = [0] + [1]*len(tokens_a) + [0] + [1]*len(tokens_b) + [0]

      instance = TrainingInstance(
          tokens=tokens,
          segment_ids=segment_ids,
          mention_id=mention_id,
          word_ids=word_ids)
      instances.append(instance)

  return instances


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(
      instances, tokenizer, FLAGS.max_seq_length, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
