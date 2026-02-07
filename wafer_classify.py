## wafer classification algorithm 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
#DATA_PATH = "data/Wafer_Map_Datasets.npz"
DATA_PATH="D:/College_event/Hackthon/Wafer_Map_Datasets.npz"
OUTPUT_DIR = "D:/College_event/Hackthon/outputs"
IMG_SIZE = (52, 52)
NUM_CLASSES = 8        # clean + selected single-defect classes (+ optional other)
BATCH_SIZE = 64
EPOCHS = 20
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1) LOAD & PREPROCESS DATA
# -------------------------------
def load_dataset(npz_path):
    data = np.load(npz_path)
    X = data["arr_0"]          # shape: (N, 52, 52)
    y_onehot = data["arr_1"]   # shape: (N, 8)
    y = np.argmax(y_onehot, axis=1)

    # Normalize and add channel dimension
    X = X.astype("float32")
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    X = X[..., np.newaxis]     # (N, 52, 52, 1)

    return X, y

X, y = load_dataset(DATA_PATH)

# Train / Val / Test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
)

# -------------------------------
# 2) MODEL DEFINITION (Lightweight CNN)
# -------------------------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = build_model((52, 52, 1), NUM_CLASSES)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# 3) TRAINING
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

model.save(os.path.join(OUTPUT_DIR, "model_tf.keras"))

# -------------------------------
# 4) INFERENCE & EVALUATION
# -------------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)
cm = confusion_matrix(y_test, y_pred)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"Precision : {prec:.4f}\n")
    f.write(f"Recall    : {rec:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"Model size (params): {model.count_params()}\n")
    f.write("Algorithm: Lightweight CNN\n")
    f.write("Framework: TensorFlow (training), ONNX (deployment)\n")
    f.write("Platform : CPU (training & inference)\n")

# Plot confusion matrix
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# -------------------------------
# 5) EXPORT TO ONNX (Edge-ready)
# -------------------------------
# pip install tf2onnx onnx
import tf2onnx

spec = (tf.TensorSpec((None, 52, 52, 1), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=os.path.join(OUTPUT_DIR, "model.onnx")
)

print("Pipeline completed successfully.")


