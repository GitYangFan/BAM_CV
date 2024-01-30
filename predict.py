import numpy as np
import tensorflow as tf
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import seaborn as sns
import data_loader
import helpers_BAM as h
import custom_layers_BAM as cl

tf.keras.backend.set_floatx('float32')
ddtype = tf.float32

# load the pretrained model
# model = tf.keras.models.load_model('./model/BAM_last.hd5')
model = tf.keras.models.load_model('./model/BAM_best.hd5')

# pixels, classes_true = data_loader.load_test_set('./dataset/test_short.csv')


# the switch function for selecting test dataset
def switch_data(case_value):
    if case_value == 1:
        img_folder = './dataset/fer2013/train'
        csv_folder = './dataset/fer2013/train_label.csv'
    elif case_value == 2:
        img_folder = './dataset/fer2013/val'
        csv_folder = './dataset/fer2013/val_label.csv'
    elif case_value == 3:
        img_folder = './dataset/fer2013/test'
        csv_folder = './dataset/fer2013/test_label.csv'
    else:
        img_folder = './dataset/fer2013/train_debug'
        csv_folder = './dataset/fer2013/train_label_debug.csv'
    return img_folder, csv_folder


img_folder, csv_folder = switch_data(3)

classes_true, names = data_loader.load_label(csv_folder)
pixels = data_loader.load_img(img_folder, names, 0, len(classes_true))
classes_pred = []

pixels_array = np.array(pixels, dtype=np.float32)
image_height, image_width = 48, 48
pixels_array = pixels_array.reshape((len(pixels_array), image_height, image_width))
predictions = model.predict(pixels_array)
for prediction in predictions:
    predicted_class = np.argmax(prediction)  # find the most possible class for each image
    print('possibility:', prediction, 'class:', predicted_class)
    classes_pred.append(predicted_class)

# for img in pixels:
#     img_array = np.array(img, dtype=np.float32)
#     image_height, image_width = 48, 48
#     img_array = img_array.reshape((1, image_height, image_width))
#     prediction = model.predict(img_array)
#     # prediction_sum = np.sum(prediction, axis=0)     # sum the first dimension of tensor (48,1,7)
#     predicted_class = np.argmax(prediction[0])    # find the most possible class
#     print('possibility:', prediction[0])
#     print('class:', predicted_class)
#     # predicted_class = np.argmax(prediction[0][0][0])
#     # print(prediction.shape)
#     # prediction_flatten = prediction.flatten()
#     # predicted_class = np.argmax(prediction_flatten)
#     # print('class:', prediction_flatten[predicted_class])
#     classes_pred.append(predicted_class)

print('classes_pred:', classes_pred)
print('classes_true:', classes_true)

# predictions = model.predict(pixels)

# ---------------------- evaluation ------------------------

# print(len(classes_true))
# print(len(classes_pred))
print('accuracy:', sk.accuracy_score(classes_true, classes_pred))
print('precision:', sk.precision_score(classes_true, classes_pred, average='macro'))
print('recall:', sk.recall_score(classes_true, classes_pred, average='macro'))
print('f1-score:', sk.f1_score(classes_true, classes_pred, average='macro'))

confusion_matrix = sk.confusion_matrix(classes_true, classes_pred)
num_class = confusion_matrix.sum(axis=1, keepdims=True)
print('num_class:', num_class.T)
confusion_matrix_prop = confusion_matrix / num_class.astype(float)   # compute the proportion of correct predictions for each class

classes = ['0=Angry', '1=Disgust', '2=Fear', '3=Happy', '4=Sad', '5=Surprise', '6=Neutral']
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix_prop, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (proportion)')
plt.subplots_adjust(wspace=0.2)
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (count)')

plt.savefig('confusion_matrix.png')
# plt.show()
