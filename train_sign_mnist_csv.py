import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# =========== PATH CONFIG ===========
TRAIN_CSV = r"D:/Python/data/sign_mnist_train.csv"
TEST_CSV  = r"D:/Python/data/sign_mnist_test.csv"

# =========== HYPERPARAMS ===========
IMG_H, IMG_W, IMG_C = 28, 28, 1
BATCH_SIZE = 128
EPOCHS = 12
LR = 1e-3
SEED = 42

def load_sign_mnist_csv(csv_path):
    df = pd.read_csv(csv_path)
    y = df["label"].values.astype("int32")
    X = df.drop(columns=["label"]).values.astype("float32")
    X = X.reshape(-1, IMG_H, IMG_W, IMG_C) / 255.0
    return X, y

X_train_raw, y_train_raw = load_sign_mnist_csv(TRAIN_CSV)
X_test_raw,  y_test_raw  = load_sign_mnist_csv(TEST_CSV)

# ===== Reindex labels to contiguous 0..(K-1) =====
unique_labels = sorted(set(y_train_raw.tolist()) | set(y_test_raw.tolist()))
label2new = {old:i for i, old in enumerate(unique_labels)}
y_train = np.array([label2new[v] for v in y_train_raw], dtype="int32")
y_test  = np.array([label2new[v] for v in y_test_raw],  dtype="int32")

num_classes = len(unique_labels)
print(f"Shapes: train={X_train_raw.shape}, test={X_test_raw.shape}, num_classes={num_classes}")
print("Old labels (sorted):", unique_labels[:10], "...", unique_labels[-10:])

# ===== Build id<->letter maps based on original labels =====
# original label L -> chr(ord('A') + L)
id_to_letter = {new_id: chr(ord('A') + old_id) for old_id, new_id in label2new.items()}
letter_to_id = {v: k for k, v in id_to_letter.items()}
with open("labels_map.json", "w", encoding="utf-8") as f:
    json.dump({"id_to_letter": id_to_letter, "letter_to_id": letter_to_id}, f, ensure_ascii=False, indent=2)
print("Saved labels_map.json (after reindex)")

# ===== tf.data =====
tf.random.set_seed(SEED)
train_ds = tf.data.Dataset.from_tensor_slices((X_train_raw, y_train)).shuffle(10000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test_raw,  y_test )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ===== Model =====
inputs = layers.Input(shape=(IMG_H, IMG_W, IMG_C))
x = inputs
x = layers.RandomRotation(0.05)(x)
x = layers.RandomZoom(0.1)(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("sign_mnist_best.keras", monitor="val_accuracy", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=test_ds,   # আলাদা val না থাকলে test-ই দিচ্ছি
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

print("\nFinal evaluation on test set:")
model.evaluate(test_ds, verbose=2)

model.save("sign_mnist_model.h5")
print("Saved model -> sign_mnist_model.h5 (best -> sign_mnist_best.keras)")
