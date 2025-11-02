import os, json
import tensorflow as tf
from tensorflow.keras import layers, models

# ====== কনফিগ ======
DATA_DIR = "data/asl"     # উপরোক্ত স্ট্রাকচার অনুযায়ী রুট ফোল্ডার
IMG_SIZE = (64, 64)       # চাইলে 96/128 দিতে পারো
BATCH_SIZE = 64
EPOCHS = 10
COLOR_MODE = "rgb"        # ডেটা গ্রেস্কেল হলে "grayscale"

AUTOTUNE = tf.data.AUTOTUNE

# ====== ডেটা লোডিং (Folder-based) ======
# যদি আলাদা train/val ফোল্ডার থাকে:
train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")
test_dir  = os.path.join(DATA_DIR, "test")

def make_ds(directory, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode=COLOR_MODE,
        shuffle=shuffle
    )

has_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)
if has_split:
    train_ds = make_ds(train_dir, shuffle=True)
    val_ds   = make_ds(val_dir, shuffle=False)
    test_ds  = make_ds(test_dir, shuffle=False) if os.path.isdir(test_dir) else None
    class_names = train_ds.class_names
else:
    # যদি শুধু data/asl/ এর ভেতরে ক্লাস ফোল্ডার থাকে, এখান থেকেই split করি
    full_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode=COLOR_MODE,
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        color_mode=COLOR_MODE,
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        seed=42
    )
    train_ds = full_ds
    test_ds = None
    class_names = train_ds.class_names

num_classes = len(class_names)
print("Detected classes:", class_names, "=>", num_classes)

# Prefetch/Cache
def optimize(ds, cache=True, shuffle=False):
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.prefetch(AUTOTUNE)

train_ds = optimize(train_ds, cache=True, shuffle=True)
val_ds   = optimize(val_ds, cache=True)
if test_ds is not None:
    test_ds  = optimize(test_ds, cache=True)

# ====== Data Augmentation ======
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# ====== মডেল ======
inputs = layers.Input(shape=(*IMG_SIZE, 1 if COLOR_MODE=="grayscale" else 3))
x = layers.Rescaling(1/255.0)(inputs)
x = data_augmentation(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ====== কলব্যাক ======
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("asl_best.keras", save_best_only=True),
]

# ====== ট্রেনিং ======
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ====== ইভ্যালুয়েশন ======
if test_ds is not None:
    model.evaluate(test_ds, verbose=2)

# ====== সেভ ======
model.save("asl_model.h5")
with open("asl_labels.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("Saved model to asl_model.h5 and labels to asl_labels.json")
