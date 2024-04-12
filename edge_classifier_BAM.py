import time

import tensorflow as tf

import custom_layers_BAM_new as cl
import numpy as np
import generator_cheby_BAM as genC
import generator_image
import helpers_BAM as h
import helpers_BAM_generalized as hg
import data_loader

start_time = time.time()

tf.keras.backend.set_floatx('float32')
ddtype = tf.float32

num_class = 7
model = cl.model_attention_final(n_channels_main=100, data_layers=0, cov_layers=2, inner_channels=256, N_exp=3,
                                 N_heads=1, num_classes=num_class)      # Note: n_channels_main must be an integer multiple of N_heads

# inputs = tf.keras.Input((None, None))
inputs = tf.keras.Input((100, 100), batch_size=128)
# inputs = tf.keras.Input((48, 48))
# inputs = tf.keras.Input(shape=(48, 48), batch_size=32)
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
    optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=5e-5),
    metrics=['accuracy']
)



def scheduler(epoch, lr):
    return lr * (1 / 10) ** (1 / 500)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


# load the data from fer2013 dataset
# train_folder = './dataset/fer2013/train'
# train_csv_folder = './dataset/fer2013/train_label.csv'
# train_labels_list, train_names = data_loader.load_label(train_csv_folder, label='emotion')
#
# val_folder = './dataset/fer2013/val'
# val_csv_folder = './dataset/fer2013/val_label.csv'
# val_labels_list, val_names = data_loader.load_label(val_csv_folder, label='emotion')
#
# debug_folder = './dataset/fer2013/train_debug'
# debug_csv_folder = './dataset/fer2013/train_label_debug.csv'
# debug_labels_list, debug_names = data_loader.load_label(debug_csv_folder, label='emotion')

# load the data from RAF-DB dataset
train_folder = './dataset/RAF-DB/aligned'
train_csv_folder = './dataset/RAF-DB/train_label_shuffled_aligned_idx0.csv'
train_labels_list, train_names = data_loader.load_label(train_csv_folder, label='emotion')

val_folder = './dataset/RAF-DB/aligned'
val_csv_folder = './dataset/RAF-DB/val_label_shuffled_aligned_idx0.csv'
val_labels_list, val_names = data_loader.load_label(val_csv_folder, label='emotion')

# load the data from wiki dataset
# train_folder = './dataset/wiki_crop/image'
# train_csv_folder = './dataset/wiki_crop/wiki_train.csv'
# train_labels_list, train_names = data_loader.load_label(train_csv_folder, label='gender')
#
# val_folder = './dataset/wiki_crop/image'
# val_csv_folder = './dataset/wiki_crop/wiki_val.csv'
# val_labels_list, val_names = data_loader.load_label(val_csv_folder, label='gender')

# create a gradient viewer callback
# gradient_callback = hg.GradientCallback(generator_image.DataGenerator_image(debug_folder, debug_csv_folder, debug_names, batch_size=21, num_classes=num_class))

# create a tensorboard callback
# to open the tensorboard, in the terminal: tensorboard --logdir=logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=1,
    update_freq='batch',
)

# create a EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     # monitering the loss on the validation set
    patience=50,           # stop when the val_loss not decreases in the last 100 epoch
    restore_best_weights=True
)

# create a CheckPoint callback
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model/BAM_best.hd5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    save_weights_only=False  # True-only save the weights, False-save the whole model
)

modell.summary()

# spe = 128
# ep = 500
spe = 64
ep = 1000

history = modell.fit(
    generator_image.DataGenerator_image(train_folder, train_labels_list, train_names, batch_size=128, num_classes=num_class),
    validation_data=generator_image.DataGenerator_image(val_folder, val_labels_list, val_names, batch_size=128, num_classes=num_class),
    epochs=ep, steps_per_epoch=spe, callbacks=[lr_scheduler, tensorboard_callback, early_stopping, checkpoint], verbose=True)

end_time = time.time()

# Compute the elapsed time
elapsed_time = end_time - start_time
# print(modell.variables)

modell.save("./model/BAM_last.hd5")
np.save("BAM_history", history.history)
model.save_weights("BAM_weights")
with open("runtime_BAM.txt", "w") as file:
    file.write(str(elapsed_time))
print('pretrained model saved! ')