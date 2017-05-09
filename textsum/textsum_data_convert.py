"""Example of Converting TextSum model data.
Usage:
python textsum_data_convert.py --command text_to_binary --in_directories dailymail/stories --out_files dailymail-train.bin,dailymail-validation.bin,dailymail-test.bin --split 0.8,0.15,0.05
python textsum_data_convert.py --command text_to_vocabulary --in_directories cnn/stories,dailymail/stories --out_files vocab
"""

import collections
import struct
import sys

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize

import tensorflow as tf
from tensorflow.core.example import example_pb2

from numpy.random import seed as random_seed
from numpy.random import shuffle as random_shuffle

random_seed(123)  # Reproducibility

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'text_to_binary',
                           'Either text_to_vocabulary or text_to_binary.'
                           'Specify FLAGS.in_directories accordingly.')
tf.app.flags.DEFINE_string('article_directories', '', 'path to title article directory')
tf.app.flags.DEFINE_string('title_directories', '', 'path to main article directory')
tf.app.flags.DEFINE_string('in_directories', '', 'path to main article directory')
tf.app.flags.DEFINE_string('out_files', '', 'comma separated paths to files')
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of data')

def _text_to_binary(article_directory, title_directory, output_filenames, split_fractions):
  filenames = _get_filenames(article_directory, title_directory)

  random_shuffle(filenames)

  start_from_index = 0
  for index, output_filename in enumerate(output_filenames):
    sample_count = int(len(filenames) * split_fractions[index])
    print(output_filename + ': ' + str(sample_count))

    end_index = min(start_from_index + sample_count, len(filenames))
    _convert_files_to_binary(filenames,start_from_index, end_index,output_filename)

    start_from_index = end_index

def _text_to_vocabulary(article_directory, vocabulary_filename, max_words=200000):
  filenames = _get_filenames(article_directory,title_directory)

  counter = collections.Counter()

  for article_filename,title_filename in filenames:
    with open(article_filename, 'r') as f:
      document = f.read()

    words = document.split()
    counter.update(words)

  with open(vocabulary_filename, 'w') as writer:
    for word, count in counter.most_common(max_words - 2):
      writer.write(word + ' ' + str(count) + '\n')
    writer.write('<s> 0\n')
    writer.write('</s> 0\n')
    writer.write('<UNK> 0\n')
    writer.write('<PAD> 0\n')

def _get_filenames(article_directory, title_directory):
  filenames = []
  for directory_name in input_directories:
    filenames.extend([(join(article_directory, f),join(title_directory, f)) for f in listdir(article_directory)])
  return filenames

def _convert_files_to_binary(input_filenames, start_index, end_index, output_filename):
  article_filenames = input_filenames[0][start_index:end_index]
  title_filenames = input_filenames[1][start_index:end_index]

  with open(output_filename, 'wb') as writer:
    for afilename,tfilename in zip(article_filenames,title_filenames):
      with open(afilename, 'r') as f:
        document = '.\n'.join(f.read().strip().split('\n')[0:2]) # first two sentences
      with open(tfilename, 'r') as f:
        title = f.read().strip()

      #document_parts = document.split('\n', 1)
      #assert len(document_parts) == 2

      title = '<d><p><s>' + title + '</s></p></d>'

      body = document_parts[1].decode('utf8').replace('\n', ' ').replace('\t', ' ')
      sentences = sent_tokenize(body)
      body = '<d><p>' + ' '.join(['<s>' + sentence + '</s>' for sentence in sentences]) + '</p></d>'
      body = body.encode('utf8')

      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([body])
      tf_example.features.feature['abstract'].bytes_list.value.extend([title])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

def main(unused_argv):
  assert FLAGS.command and FLAGS.in_directories and FLAGS.out_files
  output_filenames = FLAGS.out_files.split(',')
  article_directory,title_directory = FLAGS.in_directories.split(',')

  if FLAGS.command == 'text_to_binary':
    assert FLAGS.split

    split_fractions = [float(s) for s in FLAGS.split.split(',')]

    assert len(output_filenames) == len(split_fractions)

    _text_to_binary(article_directory,title_directory, output_filenames, split_fractions)

  elif FLAGS.command == 'text_to_vocabulary':
    assert len(output_filenames) == 1

    _text_to_vocabulary(article_directory, output_filenames[0])

if __name__ == '__main__':
  tf.app.run()


#python textsum_data_convert.py \
#  --command text_to_vocabulary \
#  --in_directories  /home/ubuntu/namas/working_2010_sent/line_article,/home/ubuntu/namas/working_2010_sent/title\
#  --out_files data/custom-vocab

#python textsum_data_convert.py --command text_to_binary --in_directories /home/ubuntu/namas/working_2010_sent/line_article,/home/ubuntu/namas/working_2010_sent/title --out_files data/2010-train.bin,data/2010-validation.bin,data/2010-test.bin --split 0.8,0.15,0.05
