import time

import tensorflow as tf

import custom_layers_BAM as cl
import numpy as np
import generator_cheby_BAM as genC
import generator_image
import helpers_BAM as h

start_time = time.time()

tf.keras.backend.set_floatx('float32')
ddtype = tf.float32

model = cl.model_attention_final(n_channels_main=100, data_layers=10, cov_layers=10, inner_channels=100, N_exp=3,
                                 N_heads=5)

inputs = tf.keras.Input((None, None))
outputs = model(inputs)
modell = tf.keras.Model(inputs, outputs)

modell.compile(
    loss=h.my_loss_categorical_penalty,
    optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=0.0005),
    metrics=[h.my_accuracy_categorical_N_d_d_c, h.my_accuracy_categorical_N_d_d_c_binary, h.my_loss_categorical_N_d_d_c,
             h.my_loss_categorical_N_d_d_c_binary,
             h.my_penalty_metric, h.precisionBinary, h.recallBinary, h.aucBinary]
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

spe = 1
ep = 3

pixels, emotion = generator_image.load_image('./dataset/train_short.csv')

modell.summary()

history = modell.fit(
    generator_image.DataGenerator_image(pixels, emotion, batch_size=32),
    epochs=ep, steps_per_epoch=spe, callbacks=[lr_scheduler], verbose=True)

end_time = time.time()

# Compute the elapsed time
elapsed_time = end_time - start_time
# print(modell.variables)

modell.save("BAM.hd5")
np.save("BAM_history", history.history)
model.save_weights("BAM_weights")
with open("runtime_BAM.txt", "w") as file:
    file.write(str(elapsed_time))
