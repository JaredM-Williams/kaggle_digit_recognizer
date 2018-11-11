# Jared Williams
# this is the Digit Recognizer Kaggle competition


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas

""" read and format data"""
train_data = np.loadtxt('./train.csv', skiprows=1, delimiter=',')
test_data = np.loadtxt('./test.csv', skiprows=1, delimiter=',')

train_labels = train_data[:, 0]
train_images = train_data[:, 1::]

test_labels = test_data[:, 0]
test_images = test_data

print("Before: ", train_images.shape)
print("Before: ", test_images.shape)

train_images = np.reshape(train_images, (42000, 28, 28, 1))
test_images = np.reshape(test_images, (28000, 28, 28, 1))

print("After: ", train_images.shape)
print("After: ", test_images.shape)




# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "digits_training2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=(28, 28, 1)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model():
    model.fit(train_images, train_labels,
              epochs=12, callbacks=[cp_callback],
              validation_data=(test_images, test_labels))

def check_model():
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


model = create_model()
train_model()
#model.load_weights("digits_training2/cp-0010.ckpt");
#check_model()


