from typing import Any, Tuple
from datetime import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
# from keras.utils import load_img
from tensorflow.keras.preprocessing import image
# import torch
#
# from kan import KAN


CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


class TrainModel:

    def __init__(self, train_csv_path: str, test_csv_path: str, out_csv_path: str):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.out_csv_path = out_csv_path

        self.model = None
        self.probability_model = None

    def prepare_train_datasets(self):
        pass

    def load_model(self, path: str):
        pass

    def create_model(self):
        pass

    def train(self):
        pass

    def calc_and_save_result_dataset(self, *args, **kwargs):
        pass

    def main(self):
        self.create_model()
        self.train()
        if os.path.isfile(self.test_csv_path):
            self.calc_and_save_result_dataset()


class TrainModelDeep(TrainModel):

    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(path)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def prepare_train_datasets(self):
        df = pd.read_csv(self.train_csv_path)
        Y = np.array(df.iloc[:, 0])
        # X = np.array(tf.reshape(df.iloc[:, 1:], [-1, 28, 28]))
        X = np.array(tf.reshape(df.iloc[:, 1:], [-1, 28, 28, 1]))

        train_images, test_images, train_labels, test_labels = train_test_split(
            X, Y, test_size=0.1, stratify=Y, random_state=127
        )

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        return train_images, test_images, train_labels, test_labels

    @staticmethod
    def kernel_init_sharp(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[:, :, 0, 0] = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0],
             ]
        )
        return kernel

    @staticmethod
    def kernel_init_edge(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[:, :, 0, 0] = np.array(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1],
             ]
        )
        return kernel

    # @staticmethod
    # def kernel_init2(shape, dtype=None, partition_info=None):
    #     kernel = np.zeros(shape)
    #     kernel[:, :, 0, 0] = np.array([[0, 1], [0, 1]])
    #     return kernel

    def create_model(self):
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Resizing(height=10, width=10, interpolation='bilinear'),
            # tf.keras.layers.Flatten(input_shape=(10, 10)),

            # tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', data_format="channels_last"), #!!!!!!!!

            # tf.keras.layers.RandomZoom(height_factor=(0.4, 0.5), width_factor=(0.4, 0.5)),

            # tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=TrainModelDeep.kernel_init_edge, activation='relu', data_format="channels_last"),

            # tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer=TrainModelDeep.kernel_init_sharp, activation='tanh', data_format="channels_last"),


# !!!!!!!!!!!!!!!!
            # tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), activation='relu', data_format="channels_last"),
            # tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last"),
            # tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), activation='relu', data_format="channels_last"),
            #
            # # tf.keras.layers.SpatialDropout2D(rate=0.2, data_format="channels_last"),
            # tf.keras.layers.Conv2D(filters=200, kernel_size=(2, 2), activation='relu', data_format="channels_last"),
            #
            # tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"),
# !!!!!!!!!!!!!!!!!!

            tf.keras.layers.Conv2D(filters=20, kernel_size=(9, 9), activation='relu', data_format="channels_last"),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last"),
            tf.keras.layers.Conv2D(filters=20, kernel_size=(7, 7), activation='relu', data_format="channels_last"),

            # tf.keras.layers.SpatialDropout2D(rate=0.2, data_format="channels_last"),
            tf.keras.layers.Conv2D(filters=20, kernel_size=(2, 2), activation='relu', data_format="channels_last"),

            # tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"),





            # tf.keras.layers.SpatialDropout2D(rate=0.05, data_format="channels_last"),
            # tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', data_format="channels_last"),
            # tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation='relu', data_format="channels_last"),

            # tf.keras.layers.MaxPool2D(pool_size=(3, 3), data_format="channels_last"),
            # tf.keras.layers.AveragePooling2D(pool_size=(3, 3), data_format="channels_last"),
            # tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)), data_format="channels_last"),

            # tf.keras.layers.DepthwiseConv2D(kernel_size=(2, 2), activation='relu', data_format="channels_last"),

            # tf.keras.layers.AveragePooling1D(pool_size=7),


            tf.keras.layers.Flatten(input_shape=(28, 28)),
            # tf.keras.layers.PReLU(),
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.AlphaDropout(0.1),

            tf.keras.layers.Dense(10),
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train(self):
        train_images, test_images, train_labels, test_labels = self.prepare_train_datasets()
        self.model.fit(train_images, train_labels, epochs=10)
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=2)

        print('\nTest accuracy:', test_acc)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        self.model.save("model.keras")

    def calc_and_save_result_dataset(self):
        df_test = pd.read_csv(self.test_csv_path)
        # X_test = np.array(tf.reshape(df_test.iloc[:, 1:], [-1, 28, 28]))
        X_test = np.array(tf.reshape(df_test.iloc[:, 1:], [-1, 28, 28, 1]))

        predictions = self.probability_model.predict(X_test)
        result = np.apply_along_axis(np.argmax, 1, predictions)

        df_out = pd.DataFrame({"Category": result})
        df_out.to_csv(self.out_csv_path, index_label="Id")

    def predict_from_picture(self, img) -> Tuple[int, str]:
        # img = load_img(path, color_mode="grayscale", target_size=(28, 28))
        img = img.resize((28, 28))
        x = image.img_to_array(img)
        x = x.reshape(28, 28, 1)
        x = x / 255
        predictions = self.probability_model.predict(np.array([x]))
        code = np.argmax(predictions[0])
        res = CLASS_NAMES[code]
        return code, res

    def generate_pictures(self):
        train_images, test_images, train_labels, test_labels = self.prepare_train_datasets()

        for i in range(10):
            plt.figure(figsize=(10, 10))
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.savefig(f"fig_{test_labels[i]}.png")


if __name__ == "__main__":
    x = TrainModelDeep(
        train_csv_path="/data/fashion-mnist_train.csv",
        test_csv_path="/data/fashion-mnist_test.csv",
        out_csv_path=f"/data/out_{datetime.now().strftime('%Y_%m_%d')}.csv",
    )
    x.main()

    # x.load_model("model.keras")
    # x.generate_pictures()
