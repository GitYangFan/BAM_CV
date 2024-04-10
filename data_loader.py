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


def load_label(csv_folder, label='emotion'):
    # load the labels
    data = pd.read_csv(csv_folder)
    labels_list = []
    img_name = []
    for index, row in data.iterrows():
        labels_list.append(row[label])
        img_name.append(row['file'])
        # if row['gender'] == 0:       # 0 for female, 1 for male
        #     labels_list.append(0)
        # else:
        #     labels_list.append(1)
    return labels_list, img_name


def get_cov_from_img(img_array):
    mean = np.mean(img_array)
    centered_data = img_array - mean
    cov_of_img = np.cov(centered_data, rowvar=False)
    return cov_of_img


def load_img(folder, img_names, start, end, standard_size):
    # load the jpg images
    pixels_list = []
    # find the first img number
    # dot_idx = img_names[0].rfind('.')
    # first_img_idx = int(img_names[0][:dot_idx])
    # add the first img number to the start and end
    # for i in range(start+first_img_idx, end+first_img_idx):
    for i in range(start, end):
        # img_name = "{}.jpg".format(i)
        img_name = img_names[i]
        img_path = os.path.join(folder, img_name)
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                # convert to grey image
                img_grey = img.convert('L')
                # Scale the image to the standard size 48x48 by downsampling
                # standard_size = (48, 48)
                img_resized = img_grey.resize(standard_size, Image.LANCZOS)
                # convert the img to numpy array
                img_array = np.array(img_resized)
                # preprocessing
                img_array_preprocessed, img_preprocessing_array_prewhiten, img_preprocessing_crop, img_preprocessing_flip, img_preprocessing_rotate, img_preprocessing_stand = preprocessing(img_array, standard_size)
                # flatten the img pixels
                pixels = tf.reshape(img_array_preprocessed, [-1])
                pixels_list.append(pixels)

                # show the plt before and after process
                # fig, axes = plt.subplots(8, 1, figsize=(5, 15))
                # axes[0].imshow(img, cmap='gray')
                # # axes[0].set_title('raw')
                # axes[1].imshow(img_grey, cmap='gray')
                # # axes[1].set_title('gray')
                # axes[2].imshow(img_resized, cmap='gray')
                # # axes[2].set_title('resize')
                # axes[3].imshow(img_preprocessing_array_prewhiten, cmap='gray')
                # # axes[3].set_title('prewhiten')
                # axes[4].imshow(img_preprocessing_crop, cmap='gray')
                # # axes[4].set_title('rand.crop')
                # axes[5].imshow(img_preprocessing_flip, cmap='gray')
                # # axes[5].set_title('rand.flip')
                # axes[6].imshow(img_preprocessing_rotate, cmap='gray')
                # # axes[6].set_title('rand.rotate')
                # axes[7].imshow(img_preprocessing_stand, cmap='gray')
                # # axes[7].set_title('standardization')
                # for ax in axes:
                #     ax.axis('off')
                # plt.show()
        else:
            print(f"Image {img_path} not found.")

    return pixels_list



def show_image_batch(pixels_batch, label_batch):
    img_batch = [tf.reshape(img, (48, 48)).numpy() for img in pixels_batch]
    rows = 2
    cols = 2
    idx = 0
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    # show the image
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(img_batch[idx], cmap='gray')
            axes[i, j].set_title(label_batch[idx])
            axes[i, j].axis('off')
            idx = idx + 1
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('sample_img.png')



"""
-------- test ------------
"""
# --------- test the image loader ---------
# train_folder = './dataset/RAF-DB/aligned'
# train_csv_folder = './dataset/RAF-DB/train_label_shuffled_aligned_idx0.csv'
# label_list, train_names = load_label(train_csv_folder, label='emotion')
# start = 250
# end = start + 10
# pixels_batch = load_img(train_folder, train_names, start, end, (48, 48))
# label_batch = label_list[start:end]
#
# print('label_batch:', label_batch)
# show_image_batch(pixels_batch, label_batch)

# --------- test the label loader ---------
# csv_folder = './dataset/fer2013/train_label.csv'
# labels_list = load_label(csv_folder)
# print('length of labels_list:', len(labels_list))
