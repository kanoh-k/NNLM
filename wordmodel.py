import tensorflow as tf
import numpy as np
import re
import glob
import collections
import random
import pickle
import time
import datetime
import os

class LanguageModel:
    def __init__(self):
        self.corpus_files = "./corpus/newspaper_newswire/*.txt"
        self.test_files = "./corpus/non-fiction/*.txt"
        self.dictionary_filename = "./data/newspaper.dic"
        self.model_filename = "./data/newspaper.ckpt"
        self.logdir = "./log"
        self.corpus_encoding = "utf-8"
        self.hidden_layer_size = 30
        self.batch_size = 32
        self.chunk_size = 5
        self.epochs = 100
        self.learning_rate = 0.005
        self.forget_bias = 1.0
        self.unknown_word_threshold = 3 # Word is mapped to <unk> if word count is less than this value
        self.unknown_word_symbol = "<unk>"


    def build_dict(self):
        # コーパス全体を見て、単語の出現回数をカウントする
        counter = collections.Counter()
        for filename in glob.glob(self.corpus_files):
            with open(filename, "r", encoding=self.corpus_encoding) as f:
                # Word breaking
                text = f.read()
                text = text.lower().replace("\n", " ")
                text = re.sub(r"[^a-z '\-]", "", text)
                text = re.sub(r"[ ]+", " ", text)

                # Preprocessing: Ignore a word starting with '-'
                words = [word for word in text.split() if not word.startswith("-")]
                counter.update(words)

        # 出現頻度の低い単語をひとつの記号にまとめる
        word_id = 0
        dictionary = {}
        for word, count in counter.items():
            if count <= self.unknown_word_threshold:
                continue

            dictionary[word] = word_id
            word_id += 1
        dictionary[self.unknown_word_symbol] = word_id

        print("# of unique words:", len(dictionary))

        # 辞書をpickleを使って保存しておく
        with open(self.dictionary_filename, "wb") as f:
            pickle.dump(dictionary, f)
            print("Dictionary is saved to", self.dictionary_filename)

        self.dictionary = dictionary


    def load_dictionary(self):
        with open(self.dictionary_filename, "rb") as f:
            self.dictionary = pickle.load(f)
            self.input_layer_size = len(self.dictionary)
            self.output_layer_size = len(self.dictionary)
            print("Dictionary is successfully loaded")
            print("Dictionary size is:", self.input_layer_size)


    def get_word_id(self, word):
        return self.dictionary.get(word, self.dictionary[self.unknown_word_symbol])


    def generate_batch(self, isTraining=True):
        batch_x = []
        batch_y = []
        files = self.corpus_files if isTraining else self.test_files
        for filename in glob.glob(files):
            with open(filename, "r", encoding=self.corpus_encoding) as f:
                # Word breaking
                text = f.read()
                text = text.lower().replace("\n", " ")
                text = re.sub(r"[^a-z '\-]", "", text)
                text = re.sub(r"[ ]+", " ", text)

                # Preprocessing: Ignore a word starting with '-'
                words = [self.get_word_id(word) for word in text.split() if not word.startswith("-")]

                index = 0
                while index + self.chunk_size < len(words):
                    batch_x.append([[x] for x in words[index:index+self.chunk_size]])
                    batch_y.append([words[index+self.chunk_size]])

                    if len(batch_x) == self.batch_size:
                        yield batch_x, batch_y
                        batch_x, batch_y = [], []
                    index += 1

        raise StopIteration


    def inference(self, input_data):
        """
        input_data: (batch_size, chunk_size, 1)
        initial_state: (batch_size, hidden_layer_size)
        """
        hidden_w = tf.Variable(tf.truncated_normal([self.input_layer_size, self.hidden_layer_size], stddev=0.01, dtype=tf.float32))
        hidden_b = tf.Variable(tf.ones([self.hidden_layer_size], dtype=tf.float32))
        output_w = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.output_layer_size], stddev=0.01, dtype=tf.float32))
        output_b = tf.Variable(tf.ones([self.output_layer_size], dtype=tf.float32))

        input_data = tf.one_hot(input_data, depth=self.input_layer_size, dtype=tf.float32) # (batch_size, chunk_size, input_layer_size)
        input_data = tf.reshape(input_data, [-1, self.chunk_size, self.input_layer_size])
        input_data = tf.transpose(input_data, [1, 0, 2]) # (chunk_size, batch_size, input_layer_size)
        input_data = tf.reshape(input_data, [-1, self.input_layer_size]) # (chunk_size * batch_size, input_layer_size)
        input_data = tf.matmul(input_data, hidden_w) + hidden_b # (chunk_size, batch_size, hidden_layer_size)
        input_data = tf.split(0, self.chunk_size, input_data) # chunk_size * (batch_size, hidden_layer_size)

        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer_size, forget_bias=self.forget_bias)
        outputs, states = tf.nn.rnn(lstm, input_data, initial_state=lstm.zero_state(self.batch_size, tf.float32))

        # The last output is the model's output
        output = tf.matmul(outputs[-1], output_w) + output_b
        return output

    def loss(self, model, labels):
        labels = tf.one_hot(labels, depth=self.output_layer_size, dtype=tf.float32)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, labels))
        return cost

    def training(self, cost):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return optimizer

    def train(self):
        input_data = tf.placeholder(tf.int32, [None, self.chunk_size, 1])
        labels = tf.placeholder(tf.int32, [None, 1])
        initial_state = tf.placeholder(tf.float32, [None, self.hidden_layer_size])
        prediction = self.inference(input_data)
        cost = self.loss(prediction, labels)
        optimizer = self.training(cost)
        perplexity = tf.reduce_mean(tf.exp(cost))

        tf.summary.scalar("Cross entropy", cost)
        summary = tf.summary.merge_all()

        timestamp = time.time()
        dirname = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")
        dirname = os.path.join(self.logdir, dirname)

        batch1_x, batch1_y = None, None

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(dirname, sess.graph)

            for epoch in range(self.epochs):
                epoch_loss = 0
                epoch_perplexity = 0
                step = 0
                for batch_x, batch_y in self.generate_batch():
                    _, c, p = sess.run([optimizer, cost, perplexity], feed_dict={input_data: batch_x, labels: batch_y})
                    epoch_loss += c
                    epoch_perplexity += p
                    step += 1

                    if batch1_x is None:
                        batch1_x, batch1_y = batch_x, batch_y

                print("Epoch {} completed out of {}; Loss = {}; Perplexity = {}".format(epoch+1, self.epochs, epoch_loss, epoch_perplexity / step))

                summary_str = sess.run(summary, feed_dict={input_data: batch1_x, labels: batch1_y})
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

            saver = tf.train.Saver()
            saver.save(sess, self.model_filename)
            print("Trained model is saved to", self.model_filename)

    def evaluate(self):
        input_data = tf.placeholder(tf.int32, [None, self.chunk_size, 1])
        labels = tf.placeholder(tf.int32, [None, 1])
        prediction = self.inference(input_data)
        cost = self.loss(prediction, labels)
        perplexity = tf.reduce_mean(tf.exp(cost))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_filename)

            step = 0
            total_perplexity = 0
            for batch_x, batch_y in self.generate_batch():
                c, p = sess.run([cost, perplexity], feed_dict={input_data: batch_x, labels: batch_y})
                print("Batch perplexity: {}, batch_loss {}".format(p, c))
                total_perplexity += p
                step += 1

            print("Perplexity of this model:", total_perplexity / step)


def build_dict():
    lm = LanguageModel()
    lm.build_dict()

def train():
    lm = LanguageModel()
    lm.load_dictionary()
    lm.train()

def eval():
    lm = LanguageModel()
    lm.load_dictionary()
    lm.evaluate()

if __name__ == "__main__":
    build_dict()
    # train()
    # eval()
