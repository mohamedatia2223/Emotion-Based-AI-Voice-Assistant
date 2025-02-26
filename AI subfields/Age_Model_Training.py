import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to extract age from filename (first integer in the filename)
def extract_age(filename):
    return int(filename.split('_')[0])
    

# Dataset path
DATASET_DIR = "UTKFace"

# Image size and batch size
IMG_SIZE = (128, 128)  
BATCH_SIZE = 32

# Get all image paths and labels
image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.jpg') or f.endswith('.png')]
ages = [extract_age(os.path.basename(img)) for img in image_paths]

# Remove any None values in case of invalid filenames
valid_data = [(img, age) for img, age in zip(image_paths, ages) if age is not None]
image_paths, ages = zip(*valid_data)

# Split dataset (80% train, 20% test)
train_paths, test_paths, train_ages, test_ages = train_test_split(image_paths, ages, test_size=0.2, random_state=42)

# Function to preprocess images and extract labels
def custom_data_generator(image_paths, ages, batch_size):
    while True:
        indices = np.random.permutation(len(image_paths))  # Shuffle data
        for i in range(0, len(image_paths), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images, batch_labels = [], []
            
            for idx in batch_indices:
                img = cv2.imread(image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # Normalize
                
                batch_images.append(img)
                batch_labels.append(ages[idx])
            
            yield np.array(batch_images), np.array(batch_labels)

# Create train and test generators
train_generator = custom_data_generator(train_paths, train_ages, BATCH_SIZE)
test_generator = custom_data_generator(test_paths, test_ages, BATCH_SIZE)

# Load Pretrained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)  # Global Average Pooling
x = Dense(256, activation="relu")(x)  # Fully connected layer
x = Dense(128, activation="relu")(x)
x = Dense(1, activation="linear")(x)  # Output layer (1 neuron for age prediction)

# Define the model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the base model (optional)
for layer in base_model.layers:
    layer.trainable = False  # Freeze pre-trained layers

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse",  # Mean Squared Error for regression
              metrics=["mae"])  # Mean Absolute Error as an evaluation metric

# Train the model
EPOCHS = 10
model.fit(
    train_generator,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=test_generator,
    validation_steps=len(test_paths) // BATCH_SIZE,
    epochs=EPOCHS
)

# Save the trained model
model.save("Models/age_model.h5")
print("Model saved")

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(test_generator, steps=len(test_paths) // BATCH_SIZE)
print(f"Test MAE: {test_mae:.2f} years")
