import numpy as np
from image_preprocessing import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import pandas as pd


class DataGenerator_image(tf.keras.utils.Sequence):
    def __init__(self, pixels, emotion, batch_size):
        self.pixels = pixels
        self.emotion = emotion
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.pixels))

    def __len__(self):
        return int(np.ceil(len(self.pixels) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_pixels = self.pixels[start:end]
        batch_emotion = one_hot(self.emotion[start:end])
        batch_pixels = np.array(batch_pixels, dtype=np.float32)
        batch_emotion = np.array(batch_emotion, dtype=np.int32)

        image_height, image_width = 48, 48

        batch_pixels = batch_pixels.reshape((self.batch_size, image_height, image_width, 1))
        # batch_pixels = convert_to_multiple_channels(batch_pixels, 3)  # expand the channel (batch. 48, 48, 3)
        batch_emotion = batch_emotion.reshape((self.batch_size, 7, 1))  # there are 7 categories of emotions
        batch_emotion = convert_to_multiple_channels(batch_emotion, image_height)  # expand the channel (batch. 7, 48)
        tensor_pixels = tf.convert_to_tensor(batch_pixels, dtype=tf.float32)
        tensor_emotion = tf.convert_to_tensor(batch_emotion, dtype=tf.int32)



        # combine the pixels with emotion
        samples = (tensor_emotion, tensor_pixels)
        print(samples[0].shape)
        print(samples[1].shape)

        return samples


def load_image(image_directory):
    # current_dir = os.getcwd()
    # image_directory = os.path.join(current_dir, 'dataset', 'train.csv')
    dataset = pd.read_csv(image_directory)
    # print(df)
    emotion = []
    pixels = []
    for index, row in dataset.iterrows():
        emotion.append(row['emotion'])
        pixels.append(row['pixels'])
    pixels_array = []
    for img in pixels:
        img_list = img.split()
        img_array = [int(value) for value in img_list]
        img_array = np.array(img_array, dtype=np.uint8)
        # preprocessing
        image_height, image_width = 48, 48
        img_array = img_array.reshape(image_height, image_width)
        img_array_preprocessed = preprocessing(image_height, image_width, img_array)
        img_array_flattened = tf.reshape(img_array_preprocessed, [-1])
        # flattening
        pixels_array.append(img_array_flattened)
    return pixels_array, emotion


def one_hot(label):
    one_hot_labels = to_categorical(label, num_classes=7)
    # print(one_hot_labels)
    return one_hot_labels


def convert_to_multiple_channels(batch_pixels, num_channel):
    three_channel_batch = np.repeat(batch_pixels, num_channel, axis=-1)
    return three_channel_batch

# current_dir = os.getcwd()
# image_directory = os.path.join(current_dir, 'dataset', 'train.csv')
# pixels, emotion = load_image('./dataset/train_short.csv')
# emotion_one_hot = one_hot(emotion)
# batch_emotion = np.array(emotion_one_hot, dtype=np.int32)
# batch_emotion = batch_emotion.reshape((len(emotion), 1, 7))
# print(batch_emotion.shape)
# test = DataGenerator_image(pixels, emotion, batch_size=32)
# print(len(pixels))
# print(len(emotion))

# test one-hot encoder:
# pixels, emotion = load_image('./dataset/train_short.csv')
# index = 1
# batch_size = 32
# start = index * batch_size
# end = (index + 1) * batch_size
# batch_pixels = pixels[start:end]
# batch_emotion = one_hot(emotion[start:end])
# print('test finished')
