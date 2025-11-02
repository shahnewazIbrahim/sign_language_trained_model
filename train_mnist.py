import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ডেটা লোড
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# মডেল তৈরি
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')  # ২৬ অক্ষর
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ট্রেইন
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
model.save('sign_model.h5')
