import numpy as np
import tensorflow as tf
import data_loader
import helpers_BAM as h
import custom_layers_BAM as cl

tf.keras.backend.set_floatx('float32')
ddtype = tf.float32

custom_objects = {
    # 'my_loss_crossentropy_var_size': h.my_loss_crossentropy_var_size,
    # 'my_loss_Frob_var_size': h.my_loss_Frob_var_size,
    # 'my_accuracy_var_size': h.my_accuracy_var_size,
    # 'cov2cor': h.cov2cor,
    'my_loss_categorical_N_d_d_c':h.my_loss_categorical_N_d_d_c,
    'my_loss_categorical_N_d_d_c_binary':h.my_loss_categorical_N_d_d_c_binary,
    # 'my_loss_categorical_combined':h.my_loss_categorical_combined,
    'weight_binary_loss':0.8,
    # 'my_loss_categorical_combined2':h.my_loss_categorical_combined(weight_binary_loss=0.8),
    'my_accuracy_categorical_N_d_d_c':h.my_accuracy_categorical_N_d_d_c,
    'my_accuracy_categorical_N_d_d_c_binary':h.my_accuracy_categorical_N_d_d_c_binary,
    'layer_N_M_d_1_to_N_M_d_C_residual': cl.layer_N_M_d_1_to_N_M_d_C_residual,
    'layer_N_M_d_C_attention_features_for_each_sample': cl.layer_N_M_d_C_attention_features_for_each_sample,
    'layer_N_M_d_C_attention_samples_for_each_feature': cl.layer_N_M_d_C_attention_samples_for_each_feature,
    #'layer_squeeze_and_excitation_N_c_d_d': cl.layer_squeeze_and_excitation_N_c_d_d,
    'layer_N_C_d_d_bilinear_attention_cov2cor_spd':cl.layer_N_C_d_d_bilinear_attention_cov2cor_spd,
    'layer_N_C_d_d_spd_activation_scaled':cl.layer_N_C_d_d_spd_activation_scaled,
    'layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2':cl.layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2,
    'layer_channels_dense_res_N_M_d_c':cl.layer_channels_dense_res_N_M_d_c,
    # 'layer_squeeze_and_excitation_shape_information_N_c_d_d':cl.layer_squeeze_and_excitation_shape_information_N_c_d_d,
    'data_N_M_d_c_to_cov_N_c_d_d':cl.data_N_M_d_c_to_cov_N_c_d_d,
    'frob': cl.frob,
    # 'spd_sigmoid_activation_N_d_d': cl.spd_sigmoid_activation_N_d_d,
    'lam_init_eps': cl.lam_init_eps,
    # 'cov2cor_N_c_d_d': cl.cov2cor_N_c_d_d,
    'l1_constraintLessEqual': cl.l1_constraintLessEqual,
    # 'get_off_diag_var_size':h.get_off_diag_var_size,
    'get_off_diag_var_size_N_d_d_c':h.get_off_diag_var_size_N_d_d_c,
    # 'class_counts0':h.class_counts0,
    # 'class_counts1':h.class_counts1,
    # 'class_counts2':h.class_counts2,
    'my_penalty':h.my_penalty,
    'my_penalty_metric':h.my_penalty_metric,
    'precisionBinary':h.precisionBinary,
    'recallBinary':h.recallBinary,
    'aucBinary':h.aucBinary,
    'my_loss_categorical_penalty':h.my_loss_categorical_penalty
}


# load the pretrained model
model = tf.keras.models.load_model('./BAM.hd5', custom_objects=custom_objects)


pixels, emotion = data_loader.load_test_set('./dataset/test_short.csv')
predicted_classes = []

for img in pixels:
    img_array = np.array(img, dtype=np.float32)
    image_height, image_width = 48, 48
    img_array = img_array.reshape((image_height, image_width, 1))
    prediction = model.predict(img_array)
    # print(prediction[0][0][0].shape)
    predicted_class = np.argmax(prediction[0][0][0], axis=0)
    predicted_classes.append(predicted_class)

print(predicted_classes)
# predictions = model.predict(pixels)