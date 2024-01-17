import time

import tensorflow as tf

import custom_layers_BAM as cl
import numpy as np
import generator_cheby_BAM as genC
import generator_image
import helpers_BAM as h
import helpers_BAM_generalized as hg
import data_loader

start_time = time.time()

tf.keras.backend.set_floatx('float32')
ddtype = tf.float32

model = cl.model_attention_final(n_channels_main=5, data_layers=1, cov_layers=3, inner_channels=5, N_exp=3,
                                 N_heads=5)

# inputs = tf.keras.Input((None, None))
inputs = tf.keras.Input((48, 48))
outputs = model(inputs)
# print('outputs:', outputs)
modell = tf.keras.Model(inputs, outputs)

# modell.compile(
#     loss=h.my_loss_categorical_penalty,
#     optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=0.0005),
#     metrics=[h.my_accuracy_categorical_N_d_d_c, h.my_accuracy_categorical_N_d_d_c_binary, h.my_loss_categorical_N_d_d_c,
#              h.my_loss_categorical_N_d_d_c_binary,
#              h.my_penalty_metric, h.precisionBinary, h.recallBinary, h.aucBinary]
# )

modell.compile(
    loss='categorical_crossentropy',  # Use the default categorical cross-entropy loss function
    optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=0.001),
    metrics=['accuracy']
)


def scheduler(epoch, lr):
    return lr * (1 / 10) ** (1 / 500)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# spe = 1
# ep = 10
# N = 1
# M_min = 1
# M_max = 10
# d_min = 10
# d_max = 100
#
# history = modell.fit(
#     genC.DataGeneratorChebyshev(N, M_min, M_max, d_min, d_max),
#     epochs=ep, steps_per_epoch=spe, callbacks=[lr_scheduler], verbose=True)

# spe = 128
# ep = 1000
spe = 3
ep = 30

# pixels, emotion = generator_image.load_image('./dataset/train.csv')
train_folder = './dataset/fer2013/train'
train_csv_folder = './dataset/fer2013/train_label.csv'
train_labels_list, train_names = data_loader.load_label(train_csv_folder)

val_folder = './dataset/fer2013/train_debug'
val_csv_folder = './dataset/fer2013/train_label_debug.csv'
val_labels_list, val_names = data_loader.load_label(val_csv_folder)

# create a gradient viewer callback
gradient_callback = hg.GradientCallback(generator_image.DataGenerator_image(val_folder, val_labels_list, val_names, batch_size=20))

modell.summary()

history = modell.fit(
    generator_image.DataGenerator_image(train_folder, train_labels_list, train_names, batch_size=20),
    validation_data=generator_image.DataGenerator_image(val_folder, val_labels_list, val_names, batch_size=20),
    epochs=ep, steps_per_epoch=spe, callbacks=[lr_scheduler, gradient_callback], verbose=True)

end_time = time.time()

# Compute the elapsed time
elapsed_time = end_time - start_time
# print(modell.variables)

modell.save("BAM.hd5")
np.save("BAM_history", history.history)
model.save_weights("BAM_weights")
with open("runtime_BAM.txt", "w") as file:
    file.write(str(elapsed_time))
print('pretrained model saved! ')
