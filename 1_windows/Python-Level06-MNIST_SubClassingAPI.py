#【１】 半角入力になっていることを確認してください。キーボードの左上に「半角/全角」キーがあります。
#【２】 CTRLキーを押しながらF5キーを押してデバッグを開始します。
#　tensorflowのライブラリを読み込んでいます。tfと名前をつけています。
# 読み込むのに少し時間がかかります。
import tensorflow as tf

#【３】 CTRLキーを押しながらF10を押してデバッグを続けます。（CTRL+F5は最初だけです。以降は、CTRL+F10です。）
#Denseクラス、Flattenクラス、Conv2Dクラスをインポートします。
from tensorflow.keras.layers import Dense, Flatten, Conv2D

#Modelクラスをインポートします。
from tensorflow.keras import Model

#MNIST(手書き数字)のモジュールを、オブジェクトとして変数に代入します。
mnist = tf.keras.datasets.mnist

#MNIST(手書き数字)のデータを読み込みます。
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#画像のピクセル値は、0~255の数値なので、255で割ることで、0~1の値に正規化します。
x_train, x_test = x_train / 255.0, x_test / 255.0

#tensorflowライブラリがデータを読み込めるように、次元を修正します。（x_train）
x_train = x_train[..., tf.newaxis]

#tensorflowライブラリがデータを読み込めるように、次元を修正します。（x_test）
x_test = x_test[..., tf.newaxis]

#訓練用のデータを準備します。(x_trainは学習用の画像データ、y_trainは学習用の教師データです。)
#x_trainは、60000枚の画像データですが、プログラムを読んでいくには、枚数が多すぎるので、0番~9番の10枚の画像データに絞ります。
#.copy()メソッドで配列に入れなおします。
x_train = x_train[0:10,:,:,:].copy()

#y_trainは、60000個の教師データですが、プログラムを読んでいくには、個数が多すぎるので、0番~9番の10枚分の教師データに絞ります。
#.copy()メソッドで配列に入れなおします。
y_train = y_train[0:10].copy()


#tf.data.Dataset.from_tensor_slicesは、スライスで画像データのミニバッチ分を取得するイテレータオブジェクトを生成します。
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1).batch(2)

#テストデータを準備します。(x_test, y_test)
x_test = x_test[0:10,:,:,:].copy()
y_test = y_test[0:10].copy()

#tf.data.Dataset.from_tensor_slicesは、スライスで画像データのミニバッチ分を取得するイテレータオブジェクトを生成します。
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2)



class MyModel(Model):
    #サブクラスMyModelの定義です。 Modelクラスを継承して定義しています。
    
    def __init__(self):
        #__init__メソッドは、コンストラクタといってクラスがインスタンス化される時に自動実行されるメソッドです。
        #__init__メソッドは、レイヤー(層)の定義をしておきます。
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        #callメソッドは、順伝播(フォワードパス)を定義しておきます。
        # FunctionalAPIのように数珠繋ぎにして記述します。
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


#MyModelクラスのオブジェクトを生成します。
model = MyModel()


#誤差関数(=損失関数)を、オブジェクトで生成しています。
#GradientTapeブロックに記述することで、誤差関数の偏微分が計算できます。
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

#最適化手法として、Adamオブジェクトを生成しています。Adamオブジェクトの方法で、パラメータWを更新します。
optimizer = tf.keras.optimizers.Adam()

#学習時（=訓練時）の誤差
train_loss = tf.keras.metrics.Mean(name='train_loss')

#学習時（=訓練時）の正解率
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#テスト時の誤差
test_loss = tf.keras.metrics.Mean(name='test_loss')

#テスト時の正解率
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


#@tf.functionは、デバッグのために外します。
def train_step(images, labels):
    with tf.GradientTape() as tape:
        #GradientTapeの中に、誤差関数を記述しておくと、誤差関数の偏微分を計算できます。
        predictions = model(images)
        loss = loss_object(labels, predictions)

    #実際に、誤差関数の偏微分を計算しているところです。計算結果の偏微分値のベクトルは、変数gradientsに格納されます。
    gradients = tape.gradient(loss, model.trainable_variables)
    
    #重みWベクトルを更新しているところです。
    #zip関数で、偏微分値のベクトルと、重みパラメータの値のベクトルをまとめて渡し、更新します。
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #訓練時の誤差（＝損失値）、正解率を計算します。
    train_loss(loss)
    train_accuracy(labels, predictions)



#@tf.function
def test_step(images, labels):
    #モデルに画像を渡して、推論しています。predictionsは推論した結果です。
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    #テスト時の誤差（＝損失値）、正解率をを計算しています。
    test_loss(t_loss)
    test_accuracy(labels, predictions)



EPOCHS = 6

#ニューラルネットワークのモデルを学習させます。（モデル内部のパラメータを調整していきます。正しく予測できるモデルにするためです。）
#エポックのループです。１エポックは、用意した画像全部を使った１ループ分の処理です。
for epoch in range(EPOCHS):
    
    #何エポック目か。コンソール画面に出力されます。
    #epochは数値なので、str関数で文字列に変換して、'エポック目'と文字列連結します。
    print(str(epoch) + 'エポック目')
    
    #ミニバッチ分の画像とラベルをとりだして訓練するループ
    for images, labels in train_ds:
        train_step(images, labels)
        
    #ミニバッチ分の画像とラベルをとりだしてテストするループ
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    #訓練状況の出力
    print ('Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(
    epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
  
    
    #1エポック分の学習が終了したので、訓練情報をリセットします。
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


print('End')