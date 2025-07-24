import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# --- 1. Configuration & Constants ---
DATA_DIR = 'data/asl_dataset'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25 # Set a higher epoch count, EarlyStopping will find the best number

# --- 2. Professional Data Loading with Stratified Split ---
print("--- Loading and Preparing Data with Stratified Split ---")
# Manually collect all file paths and their corresponding labels
filepaths = []
labels = []
classes = sorted([d for d in os.listdir(DATA_DIR) if not d.startswith('.')])
class_map = {name: i for i, name in enumerate(classes)}

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_dir):
        filepaths.append(os.path.join(class_dir, fname))
        labels.append(class_map[class_name])

# Perform a stratified split to ensure both train and val sets have representative data
train_paths, val_paths, train_labels, val_labels = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

# Create TensorFlow datasets from the file paths
def create_dataset(paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    def load_and_preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img
    img_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    return tf.data.Dataset.zip((img_ds, label_ds))

train_ds = create_dataset(train_paths, train_labels)
val_ds = create_dataset(val_paths, val_labels)

class_names = classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(buffer_size=len(train_paths)).batch(BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Data Augmentation Layer ---
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
], name="data_augmentation")

# --- 4. Model Building ---
def build_cnn_from_scratch(input_shape, num_classes):
    """Builds a simple CNN model from scratch."""
    model = Sequential([
        Input(shape=input_shape),
        tf.keras.layers.Rescaling(1./255),
        data_augmentation,
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name="cnn_from_scratch")
    return model

def build_transfer_learning_model(input_shape, num_classes):
    """Builds a model using MobileNetV2 for transfer learning."""
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="transfer_learning_mobilenet")
    return model

# --- 5. Training and Evaluation Function ---
def train_and_evaluate(model_builder, model_name):
    """Builds, compiles, trains, and evaluates a given model."""
    print(f"\n--- Processing Model: {model_name} ---")
    
    input_shape = IMG_SIZE + (3,)
    model = model_builder(input_shape, num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Define callbacks with the new .keras format
    model_path = os.path.join('models', f'best_{model_name}.keras') # <-- CHANGE TO .keras
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    actual_epochs = len(history.history['accuracy'])
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(actual_epochs), acc, label='Training Accuracy')
    plt.plot(range(actual_epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Accuracy Curves')

    plt.subplot(1, 2, 2)
    plt.plot(range(actual_epochs), loss, label='Training Loss')
    plt.plot(range(actual_epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Loss Curves')
    
    plot_filename = os.path.join('reports', 'figures', f'{model_name}_training_curves.png')
    plt.savefig(plot_filename)
    print(f"Saved training curves to {plot_filename}")
    plt.close()

    print(f"--- Generating Reports for {model_name} ---")
    best_model = tf.keras.models.load_model(model_path) # <-- Loads .keras format
    
    y_pred_indices = []
    y_true = []
    for images, labels in val_ds:
        predictions = best_model.predict(images, verbose=0)
        y_pred_indices.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred_indices)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    cm_filename = os.path.join('reports', 'figures', f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_filename)
    print(f"Saved confusion matrix to {cm_filename}")
    plt.close()

    report = classification_report(y_true, y_pred_indices, target_names=class_names)
    print(report)
    
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_val_acc = history.history['val_accuracy'][best_epoch]
    best_val_loss = history.history['val_loss'][best_epoch]
    
    return {
        "model_name": model_name,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "classification_report": report
    }

# --- 6. Main Execution ---
if __name__ == "__main__":
    scratch_results = train_and_evaluate(build_cnn_from_scratch, "cnn_from_scratch")
    transfer_results = train_and_evaluate(build_transfer_learning_model, "transfer_learning_mobilenet")
    
    summary_path = os.path.join('reports', 'model_performance_summary.txt')
    with open(summary_path, 'w') as f:
        for results in [scratch_results, transfer_results]:
            f.write(f"--- Performance Summary for: {results['model_name']} ---\n")
            f.write(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}\n")
            f.write(f"Best Validation Loss: {results['best_val_loss']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n" + "="*60 + "\n\n")
            
    print(f"\n--- Full Performance Summary saved to {summary_path} ---")
    print("--- Project Training and Evaluation Complete ---")
