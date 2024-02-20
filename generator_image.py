import numpy as np
from image_preprocessing import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import pandas as pd
import data_loader
import imblearn as imb


class DataGenerator_image(tf.keras.utils.Sequence):
    def __init__(self, folder, labels, img_names, batch_size, num_classes=7):
        self.folder = folder
        self.labels = labels
        self.batch_size = batch_size
        self.img_names = img_names
        self.num_classes = num_classes
        # counting the total number of images in the folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        image_count = 0
        for root, dirs, files in os.walk(self.folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_count += 1
        label_length = len(self.img_names)
        print('image_count:', image_count, 'label_length:', label_length)
        self.total_size = min(image_count, label_length)
        print('size:', self.total_size)
        # self.indexes = np.arange(len(self.pixels))

    def __len__(self):
        return int(np.ceil(self.total_size / self.batch_size)-1)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        standard_size = (48, 48)  # [image_height, image_width]
        # standard_size = (100, 100)  # [image_height, image_width]
        batch_pixels = data_loader.load_img(self.folder, self.img_names, start, end, standard_size)

        # check if there is enough img in the batch
        if len(batch_pixels) != self.batch_size:
            print('there is no enough data, data index is:', start)
            end = self.total_size - 1
            start = end - self.batch_size
            batch_pixels = data_loader.load_img(self.folder, self.img_names, start, end, standard_size)
            print('replacing with the last possible batch with index:', start)

        # over sampling or under sampling to balance the batch data
        batch_pixels = np.array(batch_pixels, dtype=np.float32)
        batch_labels = self.labels[start:end]
        # batch_pixels, batch_labels = over_sampling(batch_pixels, batch_labels, self.batch_size, self.num_classes)

        batch_labels = one_hot(batch_labels, self.num_classes)
        batch_labels = np.array(batch_labels, dtype=np.int32)

        # image_height, image_width = 48, 48
        # image_height, image_width = 224, 224

        # reshape the image from vector back to square matrix
        batch_pixels = batch_pixels.reshape((-1, standard_size[0], standard_size[1]))
        batch_labels = batch_labels.reshape((-1, self.num_classes))  # there are 7 categories of emotions
        tensor_pixels = tf.convert_to_tensor(batch_pixels, dtype=tf.float32)
        tensor_labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)

        # combine the pixels with emotion
        samples = (tensor_pixels, tensor_labels)
        # print(samples[0].shape)
        # print(samples[1].shape)

        return samples


def one_hot(label, num_classes):
    one_hot_labels = to_categorical(label, num_classes)
    # print(one_hot_labels)
    return one_hot_labels


def convert_to_multiple_channels(batch_pixels, num_channel):
    three_channel_batch = np.repeat(batch_pixels, num_channel, axis=-1)
    return three_channel_batch


def over_sampling(pixels, labels, batch_size, num_classes=7):
    ros = imb.over_sampling.RandomOverSampler(random_state=42)
    pixels_resampled, labels_resampled = ros.fit_resample(pixels, labels)
    # limit the number of each class in order to fit the batch size
    unique_label = np.unique(labels_resampled)
    max_samples_per_class = batch_size // len(unique_label)
    samples_to_keep = {label: min(np.sum(labels_resampled == label), max_samples_per_class) for label in unique_label}
    pixels_resampled_limit, labels_resampled_limit = [], []
    for x, label in zip(pixels_resampled, labels_resampled):
        if samples_to_keep[label] > 0:
            pixels_resampled_limit.append(x)
            labels_resampled_limit.append(label)
            samples_to_keep[label] -= 1
    return np.array(pixels_resampled_limit), np.array(labels_resampled_limit)


def under_sampling(pixels, labels):
    rus = imb.under_sampling.RandomUnderSampler(random_state=42)
    pixels_resampled, labels_resampled = rus.fit_resample(pixels, labels)
    return pixels_resampled, labels_resampled


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

# test the over sampling
# train_folder = './dataset/fer2013/train'
# train_csv_folder = './dataset/fer2013/train_label.csv'
# train_labels_list, train_names = data_loader.load_label(train_csv_folder)
# generator = DataGenerator_image(train_folder, train_labels_list, train_names, batch_size=32)
#
# for i in range(5):
#     X_batch, y_batch = generator.__getitem__(i)
#     y_batch_int = np.argmax(y_batch.numpy(), axis=1)
#     class_counts = {label: np.sum(y_batch_int == label) for label in np.unique(y_batch_int)}
#     for label, count in class_counts.items():
#         print(f"Class {label} Count: {count}")
#     print('---------------------')