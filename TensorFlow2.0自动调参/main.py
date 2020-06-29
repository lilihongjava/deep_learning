# encoding: utf-8
"""
@author: lee
@time: 2020/6/24 17:16
@file: main.py
@desc: 
"""
import gzip

import kerastuner as kt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_data():
    path = "./data/"
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = [path + each for each in files]
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)  # uint8无符号整数(0 to 255),一个字节，一张图片256色
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)  # 图像尺寸(28*28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)  # offset=8,前8不读
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)


# 图像分类模型
def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))  # 输入“压平”，即把多维的输入一维化
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])  # accuracy，用于判断模型效果的函数
    return model


# 在运行超参数搜索之前，定义一个回调在每个训练步骤结束时。
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        print("训练完成，调用回调方法")


if __name__ == '__main__':
    #  Zalando商品图片数据集
    (img_train, label_train), (img_test, label_test) = load_data()

    # 归一化
    img_train = img_train.astype('float32') / 255.0
    img_test = img_test.astype('float32') / 255.0
    # 使用 Hyperband 算法搜索超参数
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',  # 优化的目标，验证集accuracy
                         max_epochs=10,  # 最大迭代次数
                         factor=3,
                         directory='my_dir',  # my_dir/intro_to_kt目录包含超参数搜索期间运行的详细日志和checkpoints
                         project_name='intro_to_kt')

    tuner.search(img_train, label_train, epochs=10, validation_data=(img_test, label_test),
                 callbacks=[ClearTrainingOutput()])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    model.fit(img_train, label_train, epochs=10, validation_data=(img_test, label_test))
