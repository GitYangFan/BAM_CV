import os

import numpy as np
import pandas as pd
import tensorflow as tf
from image_preprocessing import preprocessing
from PIL import Image
import matplotlib.pyplot as plt


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


def load_label(csv_folder):
    # load the labels
    data = pd.read_csv(csv_folder)
    labels_list = []
    for index, row in data.iterrows():
        labels_list.append(row['emotion'])
        # if row['gender'] == 'Male':
        #     labels_list.append(0)
        # else:
        #     labels_list.append(1)
    return labels_list

def get_cov_from_img(img_array):
    mean = np.mean(img_array)
    centered_data = img_array - mean
    cov_of_img = np.cov(centered_data, rowvar=False)
    return cov_of_img

def load_img(folder, start, end):
    # load the jpg images
    pixels_list = []
    print('\n loading image in the range of:', start, end)
    for i in range(start, end):
        img_path = os.path.join(folder, f"{i+1}.jpg")
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                # convert to grey image
                img = img.convert('L')
                # Scale the image to the standard size 48x48 by downsampling
                standard_size = (48, 48)
                img_resized = img.resize(standard_size, Image.LANCZOS)
                # convert the img to numpy array
                img_array = np.array(img_resized)
                # get the covariance matrix from img
                cov_of_img = get_cov_from_img(img_array)
                # print('cov_of_img:', cov_of_img)
                # preprocessing
                img_array_preprocessed = preprocessing(cov_of_img)
                # convert to numpy array and flatten
                pixels = tf.reshape(img_array_preprocessed, [-1])
                pixels_list.append(pixels)

                # show the plt before and after process
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # axes[0].imshow(img, cmap='gray')
                # axes[0].set_title('before')
                # axes[1].imshow(img_resized, cmap='gray')
                # axes[1].set_title('after')
                # plt.show()
        else:
            print(f"Image {img_path} not found.")

    # pixels_array = np.array(pixels_list)
    return pixels_list


"""
-------- test ------------
"""
# --------- test the image loader ---------
# folder = './dataset/fairface025/train'
# start = 1
# end = 10
# pixels_list = load_img(folder, start, end)
# print('pixels_list:', pixels_list)

# --------- test the label loader ---------
# csv_folder = './dataset/fer2013/train_label.csv'
# labels_list = load_label(csv_folder)
# print('length of labels_list:', len(labels_list))
