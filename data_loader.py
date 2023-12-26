import numpy as np
import pandas as pd
import tensorflow as tf
from image_preprocessing import preprocessing


def load_train_set(image_directory):
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


def load_test_set(image_directory):
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
