import numpy as np

import generator_cheby_BAM as genC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import pandas as pd


# class ImageDataGeneratorWrapper(tf.keras.utils.Sequence):
#     def __init__(self, image_directory, batch_size=32, image_size=(224, 224), steps_per_epoch=128):
#         self.image_directory = image_directory
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.steps_per_epoch = steps_per_epoch
#         self.image_generator = ImageDataGenerator(rescale=1./255)  # You can customize this based on your image preprocessing needs
#         self.image_data_flow = self.image_generator.flow_from_directory(
#             image_directory,
#             target_size=image_size,
#             batch_size=batch_size,
#             class_mode='categorical'  # You can change this based on your task (e.g., 'binary', 'sparse', etc.)
#         )
#
#     def __len__(self):
#         return self.steps_per_epoch
#
#     def __getitem__(self, index):
#         batch_images, batch_labels = self.image_data_flow.next()
#         return batch_images, batch_labels


# spe = 2
# ep = 1000
# N = 1
# M_min = 5
# M_max = 10
# d_min = 1
# d_max = 10
# data = genC.DataGeneratorChebyshev(N=N, M_min=M_min, M_max=M_max, d_min=d_min, d_max=d_max, steps_per_epoch=spe)
# print(data)

# current_dir = os.getcwd()
# file_path = os.path.join(current_dir, 'dataset', 'train.csv')
# image_generator = ImageDataGeneratorWrapper(image_directory=file_path, batch_size=32, image_size=(48, 48), steps_per_epoch=128)
# print(image_generator)

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
        batch_emotion = self.emotion[start:end]
        batch_pixels = np.array(batch_pixels, dtype=np.float32)
        batch_emotion = np.array(batch_emotion, dtype=np.int32)

        # preprocessing

        return batch_pixels, batch_emotion


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
        pixels_array.append(img_array)
    return pixels_array, emotion


# current_dir = os.getcwd()
# image_directory = os.path.join(current_dir, 'dataset', 'train.csv')
# pixels, emotion = load_image('./dataset/train.csv')
# print(len(pixels))
# print(len(emotion))
