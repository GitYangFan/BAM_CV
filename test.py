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

def ImageDataGenerator(image_directory):
    # current_dir = os.getcwd()
    # image_directory = os.path.join(current_dir, 'dataset', 'train.csv')
    df = pd.read_csv(image_directory)
    # print(df)
    df['pixels'] = df['pixels'].apply(lambda x: [float(num) for num in x.split()])
    pixels_np = np.array(df['pixels'].values[0], dtype=np.float32)
    pixels_tensor = tf.convert_to_tensor(pixels_np)
    emotion_np = np.array(df['emotion'].values, dtype=np.int32)
    emotion_tensor = tf.convert_to_tensor(emotion_np)
    samples_tensor = tf.stack([pixels_tensor, emotion_tensor], axis=-1)
    # samples = tf.convert_to_tensor([df.pixels[0].astype('float32'), df.emotion.astype('int32')])
    return samples_tensor