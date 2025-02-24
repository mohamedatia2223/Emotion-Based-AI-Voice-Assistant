import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

# Load pre-trained model
model_path = r'C:\Users\Asus\Desktop\c\ai\fine_tuned_emotion_modelv6.0.h5'  # Update with actual model path
model = load_model(model_path)

# Unfreeze layers for fine-tuning
model.trainable = True

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=AdamW(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Define dataset paths
dataset_path = r'C:\Users\Asus\Desktop\c\ai\DATASET2\train'  # Update with actual dataset path

# Enhanced Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset="training"
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset="validation"
)

# Define Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train Model
model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=[early_stopping, reduce_lr]
)

# Save Fine-Tuned Model
model.save(r'C:\Users\Asus\Desktop\c\ai\fine_tuned_emotion_modelv7.0.h5')

print("Fine-tuning complete! Model saved.")
