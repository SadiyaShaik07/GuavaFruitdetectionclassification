import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

# ------------------ DATASET SETUP ------------------
DATASET_DIR = "GuavaDiseaseDataset/GuavaDiseaseDataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

CLASS_LABELS = ['Anthracnose', 'Fruit Flies', 'Healthy']
NUM_CLASSES = len(CLASS_LABELS)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

# ------------------ MODEL BUILDERS ------------------
def build_custom_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_pretrained_model(base_model_fn):
    base_model = base_model_fn(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------ TRAINING & SAVING ------------------
def train_and_save_model(model, model_name):
    print(f"\n🚀 Training {model_name}...\n")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
    ]
    history = model.fit(train_generator, validation_data=valid_generator, epochs=EPOCHS, callbacks=callbacks)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name.lower().replace(' ', '_')}.keras"
    model.save(model_path)

    np.save(f"models/history_{model_name.lower().replace(' ', '_')}.npy", history.history)
    print(f"✅ Model & history saved: {model_path}\n")

# ------------------ EVALUATION ------------------
metrics = {}

def evaluate_model(model_path, model_name):
    print(f"\n📊 Evaluating {model_name}...")
    model = tf.keras.models.load_model(model_path)

    y_pred_probs = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    print("\n🔍 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    loss, acc = model.evaluate(test_generator, verbose=0)
    print(f"\n✅ Test Accuracy ({model_name}): {acc * 100:.2f}%")
    print(f"📉 Test Loss ({model_name}): {loss:.4f}")

    metrics[model_name] = {"accuracy": acc, "loss": loss}

    history_path = f"models/history_{model_name.lower().replace(' ', '_')}.npy"
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        epochs = range(1, len(history["accuracy"]) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["accuracy"], label="Train")
        plt.plot(epochs, history["val_accuracy"], label="Validation")
        plt.title(f"Accuracy - {model_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["loss"], label="Train")
        plt.plot(epochs, history["val_loss"], label="Validation")
        plt.title(f"Loss - {model_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

# ------------------ TRAIN, SAVE, EVALUATE ------------------
from tensorflow.keras.applications import VGG19, ResNet50, EfficientNetB0, DenseNet121

models_to_train = {
    "CNN Model": build_custom_cnn(),
    "VGG19 Model": build_pretrained_model(VGG19),
    "ResNet50 Model": build_pretrained_model(ResNet50),
    "EfficientNetB0 Model": build_pretrained_model(EfficientNetB0),
    "DenseNet121 Model": build_pretrained_model(DenseNet121),
}

for name, model in models_to_train.items():
    train_and_save_model(model, name)
    evaluate_model(f"models/{name.lower().replace(' ', '_')}.keras", name)

# ------------------ COMPARISON PLOTS ------------------
def plot_metrics(metric_name, title, ylabel):
    plt.figure(figsize=(8, 6))
    names = list(metrics.keys())
    values = [metrics[name][metric_name] * 100 if metric_name == "accuracy" else metrics[name][metric_name] for name in names]
    sns.barplot(x=names, y=values, palette="mako")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

plot_metrics("accuracy", "Model Comparison - Accuracy", "Accuracy (%)")
plot_metrics("loss", "Model Comparison - Loss", "Loss")

print("\n✅ Full Evaluation Completed!")
