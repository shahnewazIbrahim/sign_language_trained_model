import os, json
import numpy as np
import pandas as pd
import tensorflow as tf

# ==== CSV PATHS (নিজের লোকেশন দিন) ====
TRAIN_CSV = r"D:/Python/data/sign_mnist_train.csv"
TEST_CSV  = r"D:/Python/data/sign_mnist_test.csv"

IMG_H, IMG_W, IMG_C = 28, 28, 1
BATCH_SIZE = 128

# ==== labels_map.json লোড ====
with open("labels_map.json", "r", encoding="utf-8") as f:
    lm = json.load(f)
id_to_letter = {int(k): v for k, v in lm["id_to_letter"].items()}

# ==== CSV লোড + ট্রেনিংয়ের মতো একই reindex ====
def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    y = df["label"].values.astype("int32")
    X = df.drop(columns=["label"]).values.astype("float32").reshape(-1, IMG_H, IMG_W, IMG_C) / 255.0
    return X, y

X_tr_old, y_tr_old = load_xy(TRAIN_CSV)
X_te_old, y_te_old = load_xy(TEST_CSV)

# ট্রেনিংয়ে আমরা sorted(unique old labels) দিয়ে reindex করেছিলাম
unique_old = sorted(set(y_tr_old.tolist()) | set(y_te_old.tolist()))
old2new = {old: i for i, old in enumerate(unique_old)}

y_test = np.array([old2new[v] for v in y_te_old], dtype="int32")

test_ds = tf.data.Dataset.from_tensor_slices((X_te_old, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==== মডেল লোড (best থাকলে সেটাই, নইলে h5) ====
model_path = "sign_mnist_best.keras" if os.path.exists("sign_mnist_best.keras") else "sign_mnist_model.h5"
print("Loading model:", model_path)
model = tf.keras.models.load_model(model_path)

# ==== Evaluate ====
loss, acc = model.evaluate(test_ds, verbose=2)
print(f"Test accuracy: {acc:.4f}")

# ==== কয়েকটা স্যাম্পল প্রেডিকশন ====
probs = model.predict(X_te_old[:10])
pred_ids = probs.argmax(axis=1)
true_ids = y_test[:10]

def id2ltr(i): return id_to_letter[int(i)]
rows = []
for i, (p, t) in enumerate(zip(pred_ids, true_ids)):
    rows.append(f"{i:02d}) pred: {p:2d} -> {id2ltr(p)} | true: {t:2d} -> {id2ltr(t)}")

print("\nSample predictions:")
print("\n".join(rows))
