import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# ============================================================
# 1. PHOTO CLASSIFIER
#    human = class 0
#    non_human = class 1
# ============================================================

print("\n==============================")
print("TRAINING PHOTO CLASSIFIER")
print("==============================")

PHOTO_TRAIN_DIR = "dataset/train"
PHOTO_VALIDATION_DIR = "dataset/validation"

photo_train = tf.keras.utils.image_dataset_from_directory(
    PHOTO_TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

photo_validation = tf.keras.utils.image_dataset_from_directory(
    PHOTO_VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Photo classes:", photo_train.class_names)

photo_model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

photo_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

photo_model.fit(
    photo_train,
    validation_data=photo_validation,
    epochs=EPOCHS
)

# ============================================================
# SAVE PHOTO MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

photo_model.save("models/photo_classifier.h5")

print("\nPhoto classifier saved:")
print("models/photo_classifier.h5")


# ============================================================
# 2. TAMPER DETECTOR
#    authentic = class 0
#    tampered = class 1
# ============================================================

print("\n==============================")
print("TRAINING TAMPER DETECTOR")
print("==============================")

TAMPER_TRAIN_DIR = "dataset_tamper/train"
TAMPER_VALIDATION_DIR = "dataset_tamper/validation"

tamper_train = tf.keras.utils.image_dataset_from_directory(
    TAMPER_TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

tamper_validation = tf.keras.utils.image_dataset_from_directory(
    TAMPER_VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Tamper classes:", tamper_train.class_names)

tamper_model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

tamper_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

tamper_model.fit(
    tamper_train,
    validation_data=tamper_validation,
    epochs=EPOCHS
)

# ============================================================
# SAVE TAMPER MODEL
# ============================================================

tamper_model.save("models/tamper_detector.h5")

print("\nTamper detector saved:")
print("models/tamper_detector.h5")

# ============================================================
# COMPLETED
# ============================================================

print("\n======================================")
print("BOTH MODELS TRAINING COMPLETED!")
print("======================================")
print("1. models/photo_classifier.h5")
print("2. models/tamper_detector.h5")