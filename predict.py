import numpy as np
import tensorflow as tf
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import seaborn as sns
import data_loader
import custom_layers_BAM as cl
# import custom_layers_BAM_new as cl
import generator_image
import csv
from grad_CAM2 import grad_cam_BAM
import cv2
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)

tf.keras.backend.set_floatx('float32')
ddtype = tf.float32

# from custom_layers_BAM
custom_objects = {
    'cal_logeig': cl.cal_logeig,
    '_cal_log_cov': cl._cal_log_cov,
    'baseline': cl.baseline,
    'layer_dense': cl.layer_dense,
    'feature_fusion': cl.feature_fusion,
    'data_N_M_d_c_to_cov_N_C2_C1_C1_image': cl.data_N_M_d_c_to_cov_N_C2_C1_C1_image,
    '_cal_cov_pooling': cl._cal_cov_pooling,
    'layer_N_M_d_1_to_N_x_x_C_conv': cl.layer_N_M_d_1_to_N_x_x_C_conv,
    'layer_N_c_d_d_to_N_d_d_3_LogEig': cl.layer_N_c_d_d_to_N_d_d_3_LogEig,
    'layer_softmax2': cl.layer_softmax2,
    'layer_N_M_d_1_to_N_M_d_C_residual': cl.layer_N_M_d_1_to_N_M_d_C_residual,
    'layer_N_M_d_C_attention_features_for_each_sample': cl.layer_N_M_d_C_attention_features_for_each_sample,
    'layer_N_M_d_C_attention_samples_for_each_feature': cl.layer_N_M_d_C_attention_samples_for_each_feature,
    'layer_N_C_d_d_bilinear_attention_cov2cor_spd': cl.layer_N_C_d_d_bilinear_attention_cov2cor_spd,
    'layer_N_C_d_d_spd_activation_scaled': cl.layer_N_C_d_d_spd_activation_scaled,
    'data_N_M_d_c_to_cov_N_c_d_d': cl.data_N_M_d_c_to_cov_N_c_d_d,
    'frob': cl.frob,
    'layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2': cl.layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2,
    'layer_channels_dense_res_N_M_d_c': cl.layer_channels_dense_res_N_M_d_c,
    'lam_init_eps': cl.lam_init_eps,
    'SoftPDmax_additiveScale_N_c_d_d': cl.SoftPDmax_additiveScale_N_c_d_d,
    'l1_constraintLessEqual': cl.l1_constraintLessEqual,
    'l1_constraint_columns': cl.l1_constraint_columns,
    'MultiHeadAttention_N_M_d_C_Feature': cl.MultiHeadAttention_N_M_d_C_Feature,
    'MultiHeadAttention_N_M_d_C_Sample': cl.MultiHeadAttention_N_M_d_C_Sample,
    'MultiHeadAttention_N_C_d_d_bilinear': cl.MultiHeadAttention_N_C_d_d_bilinear,
    'matrixNormalization_N_d_d_c': cl.matrixNormalization_N_d_d_c,
    'observationalNormalization_N_M_d_c': cl.observationalNormalization_N_M_d_c
}

# from custom_layers_BAM_new
# custom_objects = {
#     'cal_logeig': cl.cal_logeig,
#     '_cal_log_cov': cl._cal_log_cov,
#     'baseline': cl.baseline,
#     'layer_dense': cl.layer_dense,
#     'layer_dense2': cl.layer_dense2,
#     'feature_fusion': cl.feature_fusion,
#     'data_N_M_d_c_to_cov_N_C2_C1_C1_image': cl.data_N_M_d_c_to_cov_N_C2_C1_C1_image,
#     '_cal_cov_pooling': cl._cal_cov_pooling,
#     'layer_N_M_d_1_to_N_x_x_C_conv': cl.layer_N_M_d_1_to_N_x_x_C_conv,
#     'layer_softmax2': cl.layer_softmax2,
#     'layer_N_M_d_1_to_N_M_d_C_residual': cl.layer_N_M_d_1_to_N_M_d_C_residual,
#     'layer_N_C_d_d_bilinear_attention_cov2cor_spd': cl.layer_N_C_d_d_bilinear_attention_cov2cor_spd,
#     'layer_N_C_d_d_spd_activation_scaled': cl.layer_N_C_d_d_spd_activation_scaled,
#     'SoftPDmax_additiveScale_N_c_d_d': cl.SoftPDmax_additiveScale_N_c_d_d,
#     'l1_constraintLessEqual': cl.l1_constraintLessEqual,
#     'l1_constraint_columns': cl.l1_constraint_columns,
#     'MultiHeadAttention_N_C_d_d_bilinear': cl.MultiHeadAttention_N_C_d_d_bilinear
# }

# load the pretrained model
model = {}
# model[0] = tf.keras.models.load_model('./model/BAM_last.hd5')
model[0] = tf.keras.models.load_model('./model/BAM_best.hd5', custom_objects=custom_objects)
# model[0] = tf.keras.models.load_model('./model1/BAM_best.hd5', custom_objects=custom_objects)
# model[1] = tf.keras.models.load_model('./model2/BAM_best.hd5', custom_objects=custom_objects)
# model[2] = tf.keras.models.load_model('./model3/BAM_best.hd5', custom_objects=custom_objects)

model[0].summary()

# pixels, classes_true = data_loader.load_test_set('./dataset/test_short.csv')


# the switch function for selecting test dataset
def switch_data(case_value):
    if case_value == 1:
        img_folder = './dataset/fer2013/test'
        csv_folder = './dataset/fer2013/test_label.csv'
        classes = ['0=Angry', '1=Disgust', '2=Fear', '3=Happy', '4=Sad', '5=Surprise', '6=Neutral']
        label = 'emotion'
    elif case_value == 2:
        img_folder = './dataset/wiki_crop/image'
        csv_folder = './dataset/wiki_crop/wiki_test.csv'
        classes = ['0=female', '1=male']
        label = 'gender'
    elif case_value == 3:
        img_folder = './dataset/RAF-DB/aligned'
        csv_folder = './dataset/RAF-DB/test_label_shuffled_aligned_idx0.csv'
        classes = ['0=Surprise', '1=Fear', '2=Disgust', '3=Happy', '4=Sad', '5=Angry', '6=Neutral']
        label = 'emotion'
    elif case_value == 4:
        img_folder = './dataset/RAF-DB/aligned'
        csv_folder = './dataset/RAF-DB/train_label_shuffled_aligned_idx0.csv'
        classes = ['0=Surprise', '1=Fear', '2=Disgust', '3=Happy', '4=Sad', '5=Angry', '6=Neutral']
        label = 'emotion'
    elif case_value == 5:
        img_folder = './dataset/RAF-DB/aligned'
        csv_folder = './dataset/RAF-DB/val_label_shuffled_aligned_idx0.csv'
        classes = ['0=Surprise', '1=Fear', '2=Disgust', '3=Happy', '4=Sad', '5=Angry', '6=Neutral']
        label = 'emotion'
    else:
        img_folder = './dataset/fer2013/train_debug'
        csv_folder = './dataset/fer2013/train_label_debug.csv'
        classes = ['0=Angry', '1=Disgust', '2=Fear', '3=Happy', '4=Sad', '5=Surprise', '6=Neutral']
        label = 'emotion'
    return img_folder, csv_folder, classes, label


img_folder, csv_folder, classes, label = switch_data(3)

# standard_size = (48, 48)  # [image_height, image_width]
standard_size = (100, 100)  # [image_height, image_width]
classes_true, names = data_loader.load_label(csv_folder, label)
# pixels = data_loader.load_img(img_folder, names, 0, 10, standard_size)
# pixels_array = np.array(pixels, dtype=np.float32)
# pixels_array = pixels_array.reshape((len(pixels_array), standard_size[0], standard_size[1]))

classes_pred = []
# Prediction start!
batch_size = 32
img_gen = generator_image.DataGenerator_image(img_folder, classes_true, names, batch_size=batch_size, num_classes=len(classes))
# evaluation = model.evaluate(img_gen)  # evaluate using model.evaluate

# Soft voting based on multiple predictions
num_test = 10
num_model = len(model)
predictions_list = []
prediction = []
for i in range(0,num_test):
    print(i+1, '/', num_test, 'run of prediction....')
    for j in range(0,num_model):
        print('predict using model', j+1)
        prediction = model[j].predict(img_gen)    # evaluate using model.predict
        predictions_list.append(prediction)
predictions = sum(predictions_list)     # Accumulate the probability of each class

len_pred = len(predictions)
classes_true_cut = classes_true[0:len_pred]
print(len_pred, 'images has been classified! ')
print('batch_size:', batch_size)
for prediction in predictions:
    predicted_class = np.argmax(prediction)     # find the most possible class for each image
    # print('possibility:', prediction, 'class:', predicted_class)
    classes_pred.append(predicted_class)

print('classes_pred:', classes_pred)
print('classes_true:', classes_true[0:len_pred])

# cam, heatmap = grad_cam(model[0], img_gen.__getitem__(0)[0][0], classes_pred, "block5_conv3")
# cv2.imwrite("gradcam.jpg", cam)

# ------------------ generation of grad_cam ------------------
"""
img_gen2 = generator_image.DataGenerator_image(img_folder, classes_true, names, batch_size=1, num_classes=len(classes))
img_start_idx = 0
for num_img in range(len_pred):
    grad_cam_BAM(model[0], img_gen2.__getitem__(img_start_idx + num_img)[0], standard_size, img_start_idx + num_img)
"""

# predictions = model.predict(pixels)

# ---------------------- evaluation ------------------------

# print(len(classes_true))
# print(len(classes_pred))
print('accuracy:', sk.accuracy_score(classes_true_cut, classes_pred))
print('precision:', sk.precision_score(classes_true_cut, classes_pred, average='macro'))
print('recall:', sk.recall_score(classes_true_cut, classes_pred, average='macro'))
print('f1-score:', sk.f1_score(classes_true_cut, classes_pred, average='macro'))

confusion_matrix = sk.confusion_matrix(classes_true_cut, classes_pred)
num_class = confusion_matrix.sum(axis=1, keepdims=True)
print('num_class:', num_class.T)
confusion_matrix_prop = confusion_matrix / num_class.astype(float)   # compute the proportion of correct predictions for each class

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

# store the prediction results into csv file
csv_file = 'predictions.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Img_idx', 'Label'])
    for i, label in enumerate(classes_pred):
        writer.writerow([i, label])
