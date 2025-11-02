# export_tflite.py
import tensorflow as tf

# 1) মডেল লোড
try:
    model = tf.keras.models.load_model("sign_mnist_best.keras")
except:
    model = tf.keras.models.load_model("sign_mnist_model.h5")

# 2) float32 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("sign_mnist_float32.tflite", "wb").write(tflite_model)
print("✅ Saved sign_mnist_float32.tflite")

# (ঐচ্ছিক) INT8 কুয়ান্টাইজড (দ্রুত/ছোট, কিন্তু ইনপুট স্কেল/জিরো-পয়েন্ট হ্যান্ডেল করতে হবে)
# কমপ্লেক্সিটি এড়াতে প্রথমে float32 ব্যবহার করো।
