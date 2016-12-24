import tensorflow as tf
import numpy as np
import re
import glob
import collections
import random
import math
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE

class Corpus:
    def __init__(self):
        # パラメータ
        self.embedding_size = 100
        self.batch_size = 8
        self.num_skips = 2
        self.skip_window = 1
        self.num_epochs = 1000
        self.learning_rate = 0.1

        self.current_index = 0
        self.words = []

        self.dictionary = {} # コーパスの単語と単語ID
        self.final_embeddings = None # 最終的なベクトル表現

    def build_dataset(self):
        new_word_id = 0
        self.words = []
        self.dictionary = {}

        # コーパスとなるファイルたちを順に読み込む
        for filename in glob.glob("./corpus/*.txt"):
            with open(filename, "r", encoding="utf-8") as f:
                # 簡単な前処理をして、小文字表記の単語がスペースで区切られた状態にする。
                text = f.read()
                text = text.lower().replace("\n", " ")
                text = re.sub(r"[^a-z '\-]", "", text)
                text = re.sub(r"[ ]+", " ", text)

                for word in text.split():
                    # 新しい単語はdictionaryに登録
                    if word.startswith("-"): continue # 簡単な前処理。"-"から始まる単語は無視。
                    if word not in self.dictionary:
                        self.dictionary[word] = new_word_id
                        new_word_id += 1
                    self.words.append(self.dictionary[word])

        # 本当は、出現頻度の低い単語を「未知語」としてひとまとめにしたほうがよいが、
        # 今回はその処理はしないことにします。。
        self.vocabulary_size = new_word_id
        print("# of distinct words:", new_word_id)
        print("# of total words:", len(self.words))

    # skip-gramのバッチを作成
    def generate_batch(self):
        """
        例えばskip_windowが1の場合は、注目している単語+前後1単語の計3単語が注目している範囲(=span)となる。

        単語が"I have pineapple"の時を例にすると、skip_window=1, num_skips=2の場合は
        "have"という単語に注目し、その時の正解データは"I"と"pineapple"の2つ(=num_skips個)となる。
        よって、戻り値のbatch, labelsはそれぞれ
        batch = ["have", "have", ...]
        labels = ["I", "pineapple"]
        というようなベクトル（を単語IDに置き換えたもの）となる。
        """
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window

        self.current_index = 0
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32) # 注目してる単語
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32) # その周辺の単語

        # 次の処理範囲分のテキストがなかったらイテレーション終了
        span = 2 * self.skip_window + 1
        if self.current_index + span >= len(self.words):
            raise StopIteration

        # 今処理している範囲をbufferとして保持する
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.words[self.current_index])
            self.current_index += 1

        # バッチサイズごとにyeildで結果を返すためのループ
        for _ in range(len(self.words) // self.batch_size):
            # 注目している単語をずらすためのループ
            for i in range(self.batch_size // self.num_skips):
                target = self.skip_window
                targets_to_avoid = [self.skip_window]
                # 注目している単語の周辺の単語用のループ
                for j in range(self.num_skips):
                    while target in targets_to_avoid:
                        target = random.randint(0, span - 1)
                    targets_to_avoid.append(target)
                    batch[i * self.num_skips + j] = buffer[self.skip_window]
                    labels[i * self.num_skips + j, 0] = buffer[target]

                # 今注目している単語は処理し終えたので、処理範囲をずらす
                buffer.append(self.words[self.current_index])
                self.current_index += 1
                if self.current_index >= len(self.words):
                    raise StopIteration
            yield batch, labels
        raise StopIteration

    def train(self):
        # 単語ベクトルの変数を用意
        embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

        # NCE用の変数
        nce_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # 教師データ
        train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

        # 損失関数
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, self.batch_size // 2, self.vocabulary_size)
        )

        # 最適化
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # For similarities
        # valid_examples = np.random.choice(100, 16, replace=False)
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        # valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        # similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 決められた回数エポックを回す
            for epoch in range(self.num_epochs):
                epoch_loss = 0
                # generate_batch()で得られたバッチに対して学習を進める
                for batch_x, batch_y in self.generate_batch():
                    _, loss_value = sess.run([optimizer, loss], feed_dict={train_inputs: batch_x, train_labels: batch_y})
                    epoch_loss += loss_value

                print("Epoch", epoch, "completed out of", self.num_epochs, "-- loss:", epoch_loss)

            # 一応モデルを保存
            saver = tf.train.Saver()
            saver.save(sess, "./corpus/model/blog.ckpt")

            # 学習済みの単語ベクトル
            self.final_embeddings = normalized_embeddings.eval() # <class 'numpy.ndarray'>

        self.plot()

        # 単語IDと学習済みの単語ベクトルを保存
        with open("./corpus/model/blog.dic", "wb") as f:
            pickle.dump(self.dictionary, f)
        print("Dictionary was saved to", "./corpus/model/blog.dic")
        np.save("./corpus/model/blog.npy", self.final_embeddings)
        print("Embeddings were saved to", "./corpus/model/blog.npy/")

    def plot(self, filename="./corpus/model/blog.png"):
        tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
        plot_only=500
        low_dim_embeddings = tsne.fit_transform(self.final_embeddings[:plot_only, :])
        reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        labels = [reversed_dictionary[i] for i in range(plot_only)]

        plt.figure(figsize=(18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embeddings[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                        xy=(x, y),
                        xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
        plt.savefig(filename)
        print("Scatter plot was saved to", filename)

corpus = Corpus()
corpus.build_dataset()
corpus.train()
