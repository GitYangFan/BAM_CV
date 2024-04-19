import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)
import cv2


def grad_cam(model, image):
    last_conv_layer = model.get_layer("model_attention_final")
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["layer_dense"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    with tf.GradientTape() as tape:
        inputs = image[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (224, 224))
    plt.imshow(image, cmap='gray')
    plt.imshow(gradcam, alpha=0.5)
    plt.savefig('grad_cam.png')


def grad_cam_BAM(model, image, img_size):
    last_conv_layer = model.get_layer("model_attention_final")
    inputs_init = tf.keras.Input((100, 100), batch_size=1)
    out_conv = last_conv_layer(inputs_init)
    last_conv_layer_model = tf.keras.Model(inputs_init, out_conv)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = model.get_layer("layer_dense")(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    with tf.GradientTape() as tape:
        # inputs = image[np.newaxis, ...]
        inputs = image
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))
    # pooled_grads = grads
    last_conv_layer_output = last_conv_layer_output.numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam_cut = gradcam[:, 0:13*13]
    gradcam_2D = gradcam_cut.reshape((13, 13))
    gradcam_plot = cv2.resize(gradcam_2D, img_size)
    plt.imshow(image[0], cmap='gray')
    plt.imshow(gradcam_plot, alpha=0.5)
    plt.savefig('grad_cam.png')


# image = np.array(load_img("./data/cat.jpg", target_size=(224, 224, 3)))
# model = ResNet50()
# model.summary()
# gradcam = grad_cam(model, image)