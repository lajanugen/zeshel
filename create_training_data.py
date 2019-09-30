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
"""Create training data TF examples."""

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

flags.DEFINE_string("documents_file", None,
                    "Path to documents json file.")

flags.DEFINE_string("mentions_file", None,
                    "Path to mentions json file.")

flags.DEFINE_string("tfidf_candidates_file", None,
                    "Path to TFIDF candidates file.")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "is_training", True,
    "Training data")

flags.DEFINE_bool(
    "split_by_domain", False,
    "Split output TFRecords by domain.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("num_cands", 64, "Number of entity candidates.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")


class TrainingInstance(object):
  """A single set of features of data."""

  def __init__(self,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               mention_id,
               mention_guid,
               cand_guids):
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.mention_id = mention_id
    self.mention_guid = mention_guid
    self.cand_guids = cand_guids

  def __str__(self):
    s = ""
    s += "input_ids: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.input_ids[:FLAGS.max_seq_length]]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids[:FLAGS.max_seq_length]]))
    s += "input_mask: %s\n" % (" ".join([str(x) for x in self.input_mask[:FLAGS.max_seq_length]]))
    s += "mention_id: %s\n" % (" ".join([str(x) for x in self.mention_id[:FLAGS.max_seq_length]]))
    s += "label_id: %d\n" % self.label_id
    s += "\n"
    return s


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    num_cands, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):

    input_ids = instance.input_ids
    input_mask = instance.input_mask
    segment_ids = instance.segment_ids
    mention_id = instance.mention_id
    label_id = instance.label_id

    assert len(input_ids) == max_seq_length*num_cands
    assert len(input_mask) == max_seq_length*num_cands
    assert len(segment_ids) == max_seq_length*num_cands
    assert len(mention_id) == max_seq_length*num_cands

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["mention_id"] = create_int_feature(mention_id)
    features["label_id"] = create_int_feature([label_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens[:FLAGS.max_seq_length]]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
                "%s: %s" % (feature_name, " ".join([str(x) for x in values[:FLAGS.max_seq_length]])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_training_instances(document_files, mentions_files, tokenizer, max_seq_length,
                              rng, is_training=True):
  """Create `TrainingInstance`s from raw text."""

  documents = {}
  for input_file in document_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        line = line.strip()
        if not line:
          break
        line = json.loads(line)
        documents[line['document_id']] = line

  mentions = []
  for input_file in mentions_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        line = line.strip()
        if not line:
          break
        line = json.loads(line)
        mentions.append(line)

  tfidf_candidates = {}
  with tf.gfile.GFile(FLAGS.tfidf_candidates_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      line = line.strip()
      if not line:
        break
      d = json.loads(line)
      tfidf_candidates[d['mention_id']] = d['tfidf_candidates']

  vocab_words = list(tokenizer.vocab.keys())

  if FLAGS.split_by_domain:
    instances = {}
  else:
    instances = []

  for i, mention in enumerate(mentions):
    instance = create_instances_from_document(
            mention, documents, tfidf_candidates, tokenizer, max_seq_length,
            vocab_words, rng, is_training=is_training)

    if instance:
      if FLAGS.split_by_domain:
        corpus = mention['corpus']
        if corpus not in instances:
          instances[corpus] = []
        instances[corpus].append(instance)
      else:
        instances.append(instance)

    if i > 0 and i % 1000 == 0:
      tf.logging.info("Instance: %d" % i)

  if is_training:
    if FLAGS.split_by_domain:
      for corpus in instances:
        rng.shuffle(instances[corpus])
    else:
      rng.shuffle(instances)

  return instances


def get_context_tokens(context_tokens, start_index, end_index, max_tokens, tokenizer):
  start_pos = start_index - max_tokens
  if start_pos < 0:
    start_pos = 0
  prefix = ' '.join(context_tokens[start_pos: start_index])
  suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])
  prefix = tokenizer.tokenize(prefix)
  suffix = tokenizer.tokenize(suffix)
  mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index+1]))

  assert len(mention) < max_tokens

  remaining_tokens = max_tokens - len(mention)
  half_remaining_tokens = int(math.ceil(1.0*remaining_tokens/2))

  mention_context = []

  if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
    prefix_len = half_remaining_tokens
  elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
    prefix_len = remaining_tokens - len(suffix)
  elif len(prefix) < half_remaining_tokens:
    prefix_len = len(prefix)

  if prefix_len > len(prefix):
    prefix_len = len(prefix)

  prefix = prefix[-prefix_len:]

  mention_context = prefix + mention + suffix
  mention_start = len(prefix)
  mention_end = mention_start + len(mention) - 1
  mention_context = mention_context[:max_tokens]
  
  assert mention_start <= max_tokens
  assert mention_end <= max_tokens

  return mention_context, mention_start, mention_end


def pad_sequence(tokens, max_len):
  assert len(tokens) <= max_len
  return tokens + [0]*(max_len - len(tokens))


def create_instances_from_document(
    mention, all_documents, tfidf_candidates, tokenizer, max_seq_length,
    vocab_words, rng, is_training=True):
  """Creates `TrainingInstance`s for a single document."""

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  mention_length = int(max_num_tokens/2)
  cand_length = max_num_tokens - mention_length

  context_document_id = mention['context_document_id']
  label_document_id = mention['label_document_id']
  start_index = mention['start_index']
  end_index = mention['end_index']

  context_document = all_documents[context_document_id]['text']

  context_tokens = context_document.split()
  extracted_mention = context_tokens[start_index: end_index+1]
  extracted_mention = ' '.join(extracted_mention)
  assert extracted_mention == mention['text']
  mention_text_tokenized = tokenizer.tokenize(mention['text'])

  mention_context, mention_start, mention_end = get_context_tokens(
      context_tokens, start_index, end_index, mention_length, tokenizer)

  mention_id = mention['mention_id']
  assert mention_id in tfidf_candidates
  cand_document_ids = tfidf_candidates[mention_id]
  if not cand_document_ids:
    return None

  num_cands = FLAGS.num_cands
  if not is_training:
    cand_document_ids = cand_document_ids[:num_cands]

  if not is_training and label_document_id not in cand_document_ids:
    return None

  cand_document_ids = [cand for cand in cand_document_ids if cand != label_document_id]
  assert label_document_id not in cand_document_ids

  while len(cand_document_ids) < num_cands:
    cand_document_ids.extend(cand_document_ids)

  cand_document_ids.insert(0, label_document_id)

  cand_document_ids = cand_document_ids[:num_cands]
  assert len(cand_document_ids) == num_cands

  label_id = None
  for i, document in enumerate(cand_document_ids):
    if document == label_document_id:
      assert label_id == None
      label_id = i

  assert label_id == 0

  instance_tokens = []
  instance_input_ids = []
  instance_segment_ids = []
  instance_input_mask = []
  instance_mention_id = []

  for cand_document_id in cand_document_ids:
    tokens_a = mention_context
    cand_document = all_documents[cand_document_id]['text']
    cand_document_truncate = ' '.join(cand_document.split()[:cand_length])
    cand_document = tokenizer.tokenize(cand_document_truncate)
    tokens_b = cand_document[:cand_length]

    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b) + 1)
    input_mask = [1]*len(input_ids)
    mention_id = [0]*len(input_ids)

    # Update these indices to take [CLS] into account
    new_mention_start = mention_start + 1
    new_mention_end = mention_end + 1

    assert tokens[new_mention_start: new_mention_end+1] == mention_text_tokenized
    for t in range(new_mention_start, new_mention_end+1):
      mention_id[t] = 1

    assert len(input_ids) <= max_seq_length

    tokens = tokens + ['<pad>'] * (max_seq_length - len(tokens))
    instance_tokens.extend(tokens)
    instance_input_ids.extend(pad_sequence(input_ids, max_seq_length))
    instance_segment_ids.extend(pad_sequence(segment_ids, max_seq_length))
    instance_input_mask.extend(pad_sequence(input_mask, max_seq_length))
    instance_mention_id.extend(pad_sequence(mention_id, max_seq_length))

  instance = TrainingInstance(
      tokens=instance_tokens,
      input_ids=instance_input_ids,
      input_mask=instance_input_mask,
      segment_ids=instance_segment_ids,
      label_id=label_id,
      mention_id=instance_mention_id,
      mention_guid=mention['mention_id'],
      cand_guids=cand_document_ids)

  return instance


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  documents_files = []
  for input_pattern in FLAGS.documents_file.split(","):
    documents_files.extend(tf.gfile.Glob(input_pattern))
  mentions_files = []
  for input_pattern in FLAGS.mentions_file.split(","):
    mentions_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in documents_files:
    tf.logging.info("  %s", input_file)
  for input_file in mentions_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      documents_files, mentions_files, tokenizer, FLAGS.max_seq_length,
      rng, is_training=FLAGS.is_training)

  tf.logging.info("*** Writing to output files ***")
  tf.logging.info("  %s", FLAGS.output_file)

  if FLAGS.split_by_domain:
    for corpus in instances:
      output_file = "%s/%s.tfrecord" % (FLAGS.output_file, corpus)
      write_instance_to_example_files(instances[corpus], tokenizer, FLAGS.max_seq_length,
                                      FLAGS.num_cands, [output_file])
  else:
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.num_cands, [FLAGS.output_file])


if __name__ == "__main__":
  flags.mark_flag_as_required("documents_file")
  flags.mark_flag_as_required("mentions_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
