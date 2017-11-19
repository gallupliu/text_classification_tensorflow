# -*- coding: utf-8 -*-
# @Time    : 17-11-19 下午7:16
# @Author  : gallup
# @Email   : gallup-liu@hotmail.com
# @File    : read_write_tfrecord.py
# @Software: PyCharm


import tensorflow as tf


PAD = "_PAD"
UNK = "_UNK"


def Q2B(uchar):
  """全角转半角"""
  inside_code = ord(uchar)
  if inside_code == 0x3000:
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0
  #转完之后不是半角字符返回原来的字符
  if inside_code < 0x0020 or inside_code > 0x7e:
    return uchar
  return chr(inside_code)


def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub('|'.join(re.escape(key) for key in repls.keys()), lambda k: repls[k.group(0)], text)


def split_sentence(txt):
  sents = re.split(r'\n|\s|;|；|。|，|\.|,|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|、|╱|∥', txt)
  sents = [c for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
  sents = list(filter(None, sents))
  return sents


def normalize_punctuation(text):
  cpun = [['	'],
          ['﹗', '！'],
          ['“', '゛', '〃', '′', '＂'],
          ['”'],
          ['´', '‘', '’'],
          ['；', '﹔'],
          ['《', '〈', '＜'],
          ['》', '〉', '＞'],
          ['﹑'],
          ['【', '『', '〔', '﹝', '｢', '﹁'],
          ['】', '』', '〕', '﹞', '｣', '﹂'],
          ['（', '「'],
          ['）', '」'],
          ['﹖', '？'],
          ['︰', '﹕', '：'],
          ['・', '．', '·', '‧', '°'],
          ['●', '○', '▲', '◎', '◇', '■', '□', '※', '◆'],
          ['〜', '～', '∼'],
          ['︱', '│', '┼'],
          ['╱'],
          ['╲'],
          ['—', 'ー', '―', '‐', '−', '─', '﹣', '–', 'ㄧ', '－']]
  epun = [' ', '!', '"', '"', '\'', ';', '<', '>', '、', '[', ']', '(', ')', '?', ':', '･', '•', '~', '|', '/', '\\', '-']
  repls = {}

  for i in range(len(cpun)):
    for j in range(len(cpun[i])):
      repls[cpun[i][j]] = epun[i]

  return replace_all(repls, text)

def read_records(index=0):
  train_queue = tf.train.string_input_producer(['./data/train.tfrecords'], num_epochs=FLAGS.epochs)
  valid_queue = tf.train.string_input_producer(['./data/valid.tfrecords'], num_epochs=FLAGS.epochs)
  queue = tf.QueueBase.from_list(index, [train_queue, valid_queue])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'sentence_lengths': tf.FixedLenFeature([FLAGS.document_size], tf.int64),
          'document_lengths': tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64),
          'text': tf.FixedLenFeature([FLAGS.document_size * FLAGS.sentence_size], tf.int64),
      })

  sentence_lengths = features['sentence_lengths']
  document_lengths = features['document_lengths']
  label = features['label']
  text = features['text']

  sentence_lengths_batch, document_lengths_batch, label_batch, text_batch = tf.train.shuffle_batch(
      [sentence_lengths, document_lengths, label, text],
      batch_size=FLAGS.batch_size,
      capacity=5000,
      min_after_dequeue=1000)

  return sentence_lengths_batch, document_lengths_batch, label_batch, text_batch



  def write_data(doc, label, out_f):
    doc = split_sentence(clean_str(doc))
    document_length = len(doc)
    sentence_lengths = np.zeros((max_doc_len,), dtype=np.int64)
    data = np.ones((max_doc_len * max_sent_len,), dtype=np.int64)
    doc_len = min(document_length, max_doc_len)

    for j in range(doc_len):
      sent = doc[j]
      actual_len = len(sent)
      pos = j * max_sent_len
      sent_len = min(actual_len, max_sent_len)
      # sentence_lengths
      sentence_lengths[j] = sent_len
      # dataset
      data[pos:pos+sent_len] = [vocab.get(sent[k], 0) for k in range(sent_len)]

    features = {'sentence_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=sentence_lengths)),
                'document_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=[doc_len])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'text': tf.train.Feature(int64_list=tf.train.Int64List(value=data))}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    out_f.write(example.SerializeToString())

  # oversampling
  with tf.python_io.TFRecordWriter(train_path) as out_f:
    train_size = max(pos_train_size, neg_train_size)
    pos_train_docs = np.random.choice(upsampling(pos_docs[:pos_train_size], train_size), train_size, replace=False)
    neg_train_docs = np.random.choice(upsampling(neg_docs[:neg_train_size], train_size), train_size, replace=False)

    print(len(pos_train_docs), len(neg_train_docs))
    for i in tqdm(range(train_size)):
      pos_row = pos_train_docs[i]
      neg_row = neg_train_docs[i]
      write_data(pos_row, 1, out_f)
      write_data(neg_row, 0, out_f)

  with tf.python_io.TFRecordWriter(valid_path) as out_f:
    valid_size = max(pos_valid_size, neg_valid_size)
    pos_valid_docs = np.random.choice(upsampling(pos_docs[pos_train_size:], valid_size), valid_size, replace=False)
    neg_valid_docs = np.random.choice(upsampling(neg_docs[neg_train_size:], valid_size), valid_size, replace=False)
    for i in tqdm(range(valid_size)):
      pos_row = pos_valid_docs[i]
      neg_row = neg_valid_docs[i]
      write_data(pos_row, 1, out_f)
      write_data(neg_row, 0, out_f)

  print('Done {} records, train {}, valid {}'.format(pos_size + neg_size,
                                                     pos_train_size + neg_train_size,
                                                     pos_valid_size + neg_valid_size))

def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)

  for line in open(input_filename, "r"):
    data = line.split(",")
    label = float(data[9])
    features = [float(i) for i in data[:9]]

    example = tf.train.Example(features=tf.train.Features(feature={
        "label":
        tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        "features":
        tf.train.Feature(float_list=tf.train.FloatList(value=features)),
    }))
    writer.write(example.SerializeToString())

  writer.close()
  print("Successfully convert {} to {}".format(input_filename,
                                               output_filename))
