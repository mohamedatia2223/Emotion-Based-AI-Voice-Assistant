import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define dataset path
dataset_path = r"C:\Users\Asus\Desktop\c\ai\dataset4"

# Improved Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       # Rotate images up to 20 degrees
    width_shift_range=0.15,  # Shift width by 15%
    height_shift_range=0.15, # Shift height by 15%
    shear_range=0.3,         # Shear transformations
    zoom_range=0.25,         # Zoom images up to 25%
    horizontal_flip=True,    # Flip images horizontally
    brightness_range=[0.7, 1.3],  # Adjust brightness
    fill_mode='nearest',     # Fill missing pixels after transformation
    validation_split=0.2
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    batch_size=64,  # Increased batch size for better generalization
    color_mode='grayscale',
    class_mode='categorical',
    subset="training"
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset="validation"
)

# Build Improved CNN Model
model = keras.Sequential([
    # First Convolution Block
    keras.layers.Conv2D(32, (3,3), activation='relu', padding="same", kernel_regularizer=l2(0.001), input_shape=(48, 48, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    # Second Convolution Block
    keras.layers.Conv2D(64, (3,3), activation='relu', padding="same", kernel_regularizer=l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    # Third Convolution Block
    keras.layers.Conv2D(128, (3,3), activation='relu', padding="same", kernel_regularizer=l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    # Fourth Convolution Block
    keras.layers.Conv2D(256, (3,3), activation='relu', padding="same", kernel_regularizer=l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    # Global Average Pooling instead of Flatten (prevents overfitting)
    keras.layers.GlobalAveragePooling2D(),

    # Fully Connected Layers
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),  # Increased dropout to avoid overfitting
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)

# Train Model
model.fit(
    train_data, 
    epochs=50,  
    validation_data=val_data,
    callbacks=[early_stopping, reduce_lr]
)

# Save Model
model.save(r"C:\Users\Asus\Desktop\c\ai\emotion_modelv5.0.h5")

print("Model training complete and saved successfully!")
