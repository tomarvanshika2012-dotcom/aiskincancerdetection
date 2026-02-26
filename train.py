import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# PATH SETUP (FIXED)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")

IMG_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]

META_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

print("Checking dataset paths...")

if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

for d in IMG_DIRS:
    if not os.path.exists(d):
        raise FileNotFoundError(f"Image folder missing: {d}")

print("All dataset paths are correct ✅")

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(META_PATH)

cancer_classes = ["mel", "bcc", "akiec"]

df["label"] = df["dx"].apply(
    lambda x: "cancer" if x in cancer_classes else "non_cancer"
)

# =========================
# FIND IMAGE PATHS
# =========================
def find_image(image_id):
    for d in IMG_DIRS:
        path = os.path.join(d, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["image_id"].apply(find_image)
df = df.dropna(subset=["image_path"])

print(f"Total valid images: {len(df)}")

# =========================
# SPLIT DATA
# =========================
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

# =========================
# DATA GENERATORS (Improved Augmentation)
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_gen = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

test_gen = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

# =========================
# COMPUTE CLASS WEIGHTS
# =========================
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# =========================
# MODEL BUILDING (Fine-Tuning Enabled)
# =========================
print("Building model...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# Fine-tune last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =========================
# TRAINING
# =========================
print("Starting training...")

history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# =========================
# EVALUATION
# =========================
loss, accuracy, auc = model.evaluate(test_gen)
print("Test Accuracy:", accuracy)
print("Test AUC:", auc)

# =========================
# SAVE MODEL
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.h5")
model.save(MODEL_PATH)

print(f"Model saved successfully at: {MODEL_PATH}")