import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from PIL import Image

# Define directories (update these paths as per your system)
image_dir_train = r'C:\Drive E\me\Final Year Project\archive csv\train'
csv_dir_train = r'C:\Drive E\me\Final Year Project\archive csv\train_csv'
image_dir_test = r'C:\Drive E\me\Final Year Project\archive csv\test'
csv_dir_test = r'C:\Drive E\me\Final Year Project\archive csv\test_csv'

# Load data function
def load_data(image_dir, csv_dir):
    data = []
    labels = []

    image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

    if len(image_paths) != len(csv_files):
        raise ValueError("Number of images and CSV files do not match")

    for img_file, csv_file in zip(image_paths, csv_files):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path)
            image = image.resize((224, 224))
            image = img_to_array(image)
            data.append(image)
        except Exception as e:
            print(f"Error loading or processing image: {img_path}, {e}")
            continue

        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)
        if 'character' not in df.columns:
            print(f"Column 'character' not found in CSV file: {csv_path}")
            continue

        label = df['character'].values[0]
        labels.append(label)

    if not labels:
        raise ValueError("No labels found. Check if the CSV files are correct and non-empty.")

    data = np.array(data, dtype="float16") / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    return data, labels

# Load data
x_train, y_train = load_data(image_dir_train, csv_dir_train)
x_test, y_test = load_data(image_dir_test, csv_dir_test)

# Number of classes
num_classes = 7

# Function to create model with Dropout and L2 regularization
def create_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_loss_per_fold = []
val_loss_per_fold = []

# Training with cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
    print(f'Fold {fold+1}')

    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    model = create_model()

    # Model checkpoint for saving the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'best_model_fold_{fold+1}.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    # Early stopping to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model 
    history = model.fit(datagen.flow(x_train_fold, y_train_fold, batch_size=32), 
                        epochs=30, 
                        validation_data=(x_val_fold, y_val_fold),
                        callbacks=[checkpoint, early_stopping])

    # Record loss per fold
    train_loss_per_fold.append(history.history['loss'])
    val_loss_per_fold.append(history.history['val_loss'])

# Plot training and validation loss
for i in range(5):
    plt.plot(train_loss_per_fold[i], label=f'Train Fold {i+1}')
    plt.plot(val_loss_per_fold[i], label=f'Val Fold {i+1}')
    plt.title(f'Fold {i+1} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Model evaluation (Load the best model from the last fold)
model = tf.keras.models.load_model(f'best_model_fold_{fold+1}.keras')

# Predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)

# Calculate precision, recall, f1-score, accuracy, and classification report
precision = precision_score(np.argmax(y_test, axis=1), predicted_classes, average='weighted')
recall = recall_score(np.argmax(y_test, axis=1), predicted_classes, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=1), predicted_classes, average='weighted')
accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_classes)
class_report = classification_report(np.argmax(y_test, axis=1), predicted_classes, target_names=[f'Class {i}' for i in range(num_classes)])

# Print all metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()