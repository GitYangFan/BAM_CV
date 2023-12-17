import os
import pdb
import pandas as pd
import generator_cheby_BAM as genC
# import generator_image as genImage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from scipy import ndimage

"""
spe = 128
ep = 1000
N = 1
M_min = 50
M_max = 1000
d_min = 10
d_max = 100
test = genC.DataGeneratorChebyshev(N=N, M_min=M_min, M_max=M_max, d_min=d_min, d_max=d_max, steps_per_epoch=spe)
# pdb.set_trace()
print(test)
"""


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def random_rotate_image(image):
    angle = np.random.uniform(low=-20.0, high=20.0)
    return ndimage.rotate(image, angle, reshape=False, mode='nearest')


def preprocessing(image_height, image_width, img_gray_array):
    # pre whiten
    img_preprocessing = prewhiten(img_gray_array)

    # random crop (95% of original size)
    # img_preprocessing = tf.image.random_crop(img_preprocessing, [int(0.95 * image_height), int(0.95 * image_width)])

    # random flip left and right
    img_preprocessing = tf.expand_dims(img_preprocessing, axis=-1)
    img_preprocessing = tf.image.random_flip_left_right(img_preprocessing)

    # random rotate the image
    img_preprocessing = random_rotate_image(img_preprocessing)

    # standardization
    img_preprocessing = tf.image.per_image_standardization(img_preprocessing)
    return img_preprocessing


def main():
    emotions = []
    pixels = []

    # load the dataset
    csv_file_path = './dataset/train.csv'
    dataset = pd.read_csv(csv_file_path)

    for index, row in dataset.iterrows():
        emotions.append(row['emotion'])
        pixels.append(row['pixels'])
        # usage.append(row['Usage'])

    # show a sample
    idx = random.randint(1, len(emotions) - 25)
    print('length: ', len(emotions))
    print("Emotions:", emotions[idx])
    print("Pixels:", pixels[idx])
    # print("Usage:", usage[idx])
    # gray_values_list = pixels[idx].split()    # choose the first picture to show
    # gray_array = [int(value) for value in gray_values_list]
    # gray_array = np.array(gray_array, dtype=np.uint8)
    # image_height, image_width = 48, 48
    # gray_array = gray_array.reshape(image_height, image_width)
    #
    rows = 4
    cols = 2
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(8, 8))
    # axes[1, 1].imshow(gray_array, cmap='gray')
    #

    for i in range(rows):
        for j in range(cols):
            img_gray_values_list = pixels[idx].split()  # choose the first picture to show
            idx = idx + 1
            img_gray_array = [int(value) for value in img_gray_values_list]
            img_gray_array = np.array(img_gray_array, dtype=np.uint8)
            image_height, image_width = 48, 48
            img_gray_array = img_gray_array.reshape(image_height, image_width)
            axes[i, 2 * j - 2].imshow(img_gray_array, cmap='gray')
            axes[i, 2 * j - 2].axis('off')

            # --------- preprocessing -------------

            # pre whiten
            img_preprocessing = prewhiten(img_gray_array)

            # random crop (95% of original size)
            img_preprocessing = tf.image.random_crop(img_preprocessing,
                                                     [int(0.95 * image_height), int(0.95 * image_width)])

            # random flip left and right
            img_preprocessing = tf.expand_dims(img_preprocessing, axis=-1)
            img_preprocessing = tf.image.random_flip_left_right(img_preprocessing)

            # random rotate the image
            img_preprocessing = random_rotate_image(img_preprocessing)

            # standardization
            img_preprocessing = tf.image.per_image_standardization(img_preprocessing)
            # -------------- end ---------------

            # show the image
            axes[i, 2 * j - 1].imshow(img_preprocessing, cmap='gray')
            axes[i, 2 * j - 1].set_title(emotions[idx])
            axes[i, 2 * j - 1].axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


# if __name__ == '__main__':
#     main()


