import numpy as np
import tensorflow as tf
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import seaborn as sns
import data_loader
import custom_layers_BAM as cl
import generator_image

tf.keras.backend.set_floatx('float32')
ddtype = tf.float32

custom_objects = {
    'weight_binary_loss':0.8,
    'layer_N_M_d_1_to_N_M_d_C_residual': cl.layer_N_M_d_1_to_N_M_d_C_residual,
    'layer_N_M_d_C_attention_features_for_each_sample': cl.layer_N_M_d_C_attention_features_for_each_sample,
    'layer_N_M_d_C_attention_samples_for_each_feature': cl.layer_N_M_d_C_attention_samples_for_each_feature,
    'layer_N_C_d_d_bilinear_attention_cov2cor_spd':cl.layer_N_C_d_d_bilinear_attention_cov2cor_spd,
    'layer_N_C_d_d_spd_activation_scaled':cl.layer_N_C_d_d_spd_activation_scaled,
    'layer_N_c_d_d_to_N_d_d_3_softmax':cl.layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2(7),
    'layer_channels_dense_res_N_M_d_c':cl.layer_channels_dense_res_N_M_d_c,
    'data_N_M_d_c_to_cov_N_c_d_d':cl.data_N_M_d_c_to_cov_N_c_d_d,
    'layer_N_C_d_d_bilinear_attention1': cl.MultiHeadAttention_N_C_d_d_bilinear(num_heads=2),
    'layer_N_C_d_d_spd_activation1': cl.layer_N_C_d_d_spd_activation_scaled(N_exp=3),
    'frob': cl.frob,
    'lam_init_eps': cl.lam_init_eps,
    'l1_constraintLessEqual': cl.l1_constraintLessEqual,
    'layer_N_M_d_1_to_N_x_x_C_conv': cl.layer_N_M_d_1_to_N_x_x_C_conv(out_filters=100),
    'layer_N_c_d_d_to_N_d_d_3_LogEig': cl.layer_N_c_d_d_to_N_d_d_3_LogEig(7),
    'layer_dense': cl.layer_dense(7),
    'layer_softmax2': cl.layer_softmax2(7),
    'layer_baseline': cl.baseline
}

# load the pretrained model
model = tf.keras.models.load_model('./model/BAM_last.hd5', custom_objects=custom_objects)
# model = tf.keras.models.load_model('./model/BAM_best.hd5', custom_objects=custom_objects)

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


img_folder, csv_folder, classes, label = switch_data(4)

# standard_size = (48, 48)  # [image_height, image_width]
standard_size = (100, 100)  # [image_height, image_width]
classes_true, names = data_loader.load_label(csv_folder, label)
pixels = data_loader.load_img(img_folder, names, 0, len(classes_true), standard_size)
classes_pred = []

pixels_array = np.array(pixels, dtype=np.float32)
pixels_array = pixels_array.reshape((len(pixels_array), standard_size[0], standard_size[1]))

# """ evaluate using model.evaluate
# img_gen = generator_image.DataGenerator_image(img_folder, classes_true, names, batch_size=128, num_classes=len(classes))
# evaluation = model.evaluate(img_gen)

# """ evaluate using model.predict
predictions = model.predict(pixels_array)
for prediction in predictions:
    predicted_class = np.argmax(prediction)     # find the most possible class for each image
    # print('possibility:', prediction, 'class:', predicted_class)
    classes_pred.append(predicted_class)

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

# """