# Character-based languaged model by RNN

import tensorflow as tf
import numpy as np
import re
import glob
import time
import datetime

class Corpus:
    def __init__(self):
        # パラメータ
        self.chunk_size = 5
        self.vocabulary_size = 27 # data_size

        # ./corpusディレクトリの下のテキストファイルをすべて読み込んで一つのテキストにまとめる
        text = ""
        for filename in glob.glob("./corpus/*.txt"):
            with open(filename, "r", encoding="utf-8") as f:
                text += f.read() + " "

        # 簡単な前処理。a-zとスペースのみの状態にする。
        text = text.lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-z ]", "", text)
        text = re.sub(r"[ ]+", " ", text)

        # 文字列をone-hot表現のベクトルの列に変換する
        self.data_num = len(text) - self.chunk_size
        self.data = self.text_to_matrix(text)


    def prepare_data(self):
        """訓練データとテストデータを用意する。
        入力データと出力データはそれぞれ次のような次元になるべき。
        入力： (data_num, chunk_size, vocabulary_size)
        出力： (data_num, vocabulary_size)
        """

        # 入力と出力の次元テンソルを用意
        all_input = np.zeros([self.chunk_size, self.vocabulary_size, self.data_num])
        all_output = np.zeros([self.vocabulary_size, self.data_num])

        # 用意したテンソルに、コーパスのone-hot表現(self.data)からデータを埋めていく
        # i番目からi + chunk_size - 1番目までの文字が1組の入力となる
        for i in range(self.data_num):
            # このときの出力はi + chunk_size番目の文字
            all_output[:, i] = self.data[:, i + self.chunk_size]
            for j in range(self.chunk_size):
                all_input[j, :, i] = self.data[:, i + self.chunk_size - j - 1]

        # 後に使うデータ形式に合わせるために転置をとる
        all_input = all_input.transpose([2, 0, 1])
        all_output = all_output.transpose()

        # 訓練データ：テストデータを4:1に分割する
        training_num = self.data_num * 4 // 5
        return all_input[:training_num], all_output[:training_num], all_input[training_num:], all_output[training_num:]

    # 1か所だけ1が立ったベクトルを返す
    def make_one_hot(self, char):
        index = self.vocabulary_size - 1 if char == " " else (ord(char) - ord("a"))
        value = np.zeros(self.vocabulary_size)
        value[index] = 1
        return value

    # テキスト中の全ての文字をone-hot表現のベクトルに変換する
    def text_to_matrix(self, text):
        data = np.array([self.make_one_hot(char) for char in text])
        return data.transpose() # (vocabulary_size, data_num)

class CharacterBasedLM:
    """
    Input layer: vocabulary_size = 27
    Hidden layer: rnn_size = 30
    Output layter: vocabulary_size = 27
    """
    def __init__(self):
        self.input_layer_size = 27  # 入力層の数
        self.hidden_layer_size = 30 # 隠れ層のRNNユニットの数
        self.output_layer_size = 27 # 出力層の数
        self.batch_size = 128       # バッチサイズ
        self.chunk_size = 5         # 展開するシーケンスの数。この値が5の場合、c_0, c_1, ..., c_4を入力し、c_5の確率が出力される。
        self.learning_rate = 0.001  # 学習率
        self.epochs = 1000          # 学習するエポック数

    def inference(self, input_data, initial_state):
        """
        input_data: (batch_size, chunk_size, input_layer_size = vocabulary_size) 次元のテンソル
        initial_state: (batch_size, hidden_layer_size) 次元の行列
        """
        # 重みとバイアスの初期化。
        hidden_w = tf.Variable(tf.truncated_normal([self.input_layer_size, self.hidden_layer_size], stddev=0.01))
        hidden_b = tf.Variable(tf.ones([self.hidden_layer_size]))
        output_w = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.output_layer_size], stddev=0.01))
        output_b = tf.Variable(tf.ones([self.output_layer_size]))

        # BasicRNNCellは(batch_size, hidden_layer_size)がchunk_sizeつながったリストを入力とします。
        # 現時点で入力データは(batch_size, chunk_size, input_layer_size)という3次元のテンソルなので、
        # tf.transposeやtf.reshapeなどを駆使してテンソルのサイズを調整してあげます。
        input_data = tf.transpose(input_data, [1, 0, 2]) # 転置。(chunk_size, batch_size, input_layer_size=vocabulary_size)
        input_data = tf.reshape(input_data, [-1, self.input_layer_size]) # 変形。(chunk_size * batch_size, input_layer_size)
        input_data = tf.matmul(input_data, hidden_w) + hidden_b # 重みWとバイアスBを適用。 (chunk_size, batch_size, hidden_layer_size)
        input_data = tf.split(0, self.chunk_size, input_data) # リストに分割。chunk_size * (batch_size, hidden_layer_size)

        # BasicRNNCellを定義して、先ほど準備した入力データを食わせます。
        cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_layer_size)
        outputs, states = tf.nn.rnn(cell, input_data, initial_state=initial_state)

        # 最後に隠れ層から出力層につながる重みとバイアスを処理して終了です。
        # 出力層はchunk_size個のベクトルを出力しますが、興味があるのは最後の1文字だけなので
        # outputs[-1] で最後の1文字だけを処理します。
        # 言語モデルなので出力層は確率で解釈したいのですが、softmax層はこの関数の外側で
        # 定義することにします。
        output = tf.matmul(outputs[-1], output_w) + output_b

        return output

    def loss(self, logits, actual_labels):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, actual_labels))
        return cost

    def training(self, cost):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return optimizer

    def train(self):
        # 変数等の用意
        input_data = tf.placeholder("float", [None, self.chunk_size, self.input_layer_size])
        actual_labels = tf.placeholder("float", [None, self.output_layer_size])
        initial_state = tf.placeholder("float", [None, self.hidden_layer_size])

        prediction = self.inference(input_data, initial_state)
        cost = self.loss(prediction, actual_labels)
        optimizer = self.training(cost)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(actual_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # TensorBoardで可視化するため、クロスエントロピーをサマリーに追加
        tf.summary.scalar("Cross entropy", cost)
        summary = tf.summary.merge_all()

        # 訓練・テストデータの用意
        corpus = Corpus()
        trX, trY, teX, teY = corpus.prepare_data()
        training_num = trX.shape[0]

        # ログを保存するためのディレクトリ
        timestamp = time.time()
        dirname = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")

        # ここから実際に学習を走らせる
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter("./log/" + dirname, sess.graph)

            # エポックを回す
            for epoch in range(self.epochs):
                step = 0
                epoch_loss = 0
                epoch_acc = 0

                # 訓練データをバッチサイズごとに分けて学習させる (=optimizerを走らせる)
                # エポックごとの損失関数の合計値や（訓練データに対する）精度も計算しておく
                while (step + 1) * self.batch_size < training_num:
                    start_idx = step * self.batch_size
                    end_idx = (step + 1) * self.batch_size

                    batch_xs = trX[start_idx:end_idx, :, :]
                    batch_ys = trY[start_idx:end_idx, :]

                    _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={input_data: batch_xs, actual_labels: batch_ys, initial_state: np.zeros([self.batch_size, self.hidden_layer_size])})
                    epoch_loss += c
                    epoch_acc += a
                    step += 1

                # コンソールに損失関数の値や精度を出力しておく
                print("Epoch", epoch, "completed ouf of", self.epochs, "-- loss:", epoch_loss, " -- accuracy:", epoch_acc / step)

                # Epochが終わるごとにTensorBoard用に値を保存
                summary_str = sess.run(summary, feed_dict={input_data: trX, actual_labels: trY, initial_state: np.zeros([trX.shape[0], self.hidden_layer_size])})
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

            # 学習したモデルも保存しておく
            saver = tf.train.Saver()
            saver.save(sess, "./data/wiki2.ckpt")

            # 最後にテストデータでの精度を計算して表示する
            a = sess.run(accuracy, feed_dict={input_data: teX, actual_labels: teY, initial_state: np.zeros([teX.shape[0], self.hidden_layer_size])})
            print("Accuracy on test:", a)

    def predict(self, context):
        """ あるコンテキストで次に来る文字の確率を予測する

        context: str, 予測したい文字の直前の文字列。chunk_size文字以上の長さが必要。
        """
        # 最初に復元したい変数をすべて定義してしまいます
        tf.reset_default_graph()
        input_data = tf.placeholder("float", [None, self.chunk_size, self.input_layer_size])
        initial_state = tf.placeholder("float", [None, self.hidden_layer_size])
        prediction = tf.nn.softmax(self.inference(input_data, initial_state))
        predicted_labels = tf.argmax(prediction, 1)

        # 入力データの作成。contextをone-hot表現に変換する
        x = np.zeros([1, self.chunk_size, self.input_layer_size])
        for i in range(self.chunk_size):
            char = context[len(context) - self.chunk_size + i]
            index = self.input_layer_size - 1 if char == " " else (ord(char) - ord("a"))
            x[0][i][index] = 1
        feed_dict = {
            input_data: x,# (1, chunk_size, vocabulary_size)
            initial_state: np.zeros([1, self.hidden_layer_size])
        }

        # tf.Session()を用意
        with tf.Session() as sess:
            # 保存したモデルをロードする。ロード前にすべての変数を用意しておく必要がある。
            saver = tf.train.Saver()
            saver.restore(sess, "./data/wiki2.ckpt")

            # ロードしたモデルを使って予測結果を計算
            u, v = sess.run([prediction, predicted_labels], feed_dict=feed_dict)

            # コンソールに文字ごとの確率を表示
            for i in range(27):
                c = "_" if i == 26 else chr(i + ord('a'))
                print(c, ":", u[0][i])

            print("Prediction:", context + ("_" if v[0] == 26 else chr(v[0] + ord('a'))))

        return u[0]

def main():
    lm = CharacterBasedLM()

    # 学習する場合はこのように呼び出す
    # lm.train()

    # 保存したモデルを使う場合の例
    # lm.predict("restauran")

if __name__ == "__main__":
    main()
