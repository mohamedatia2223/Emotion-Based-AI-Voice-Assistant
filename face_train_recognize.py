import os
import cv2
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 1) Ask User Name
# ---------------------------
user_name = input("Enter your name (no spaces): ").strip()
dataset_dir = "dataset"
user_folder = os.path.join(dataset_dir, user_name)

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# ---------------------------
# 2) Capture Images from Webcam
# ---------------------------
cap = cv2.VideoCapture(0)
num_images = 30  # Number of images to capture
count = 0
print(f"Capturing {num_images} images for user: {user_name}. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame
    cv2.imshow("Capture Images", frame)

    # Convert to grayscale (optional, but let's keep color for training)
    # Wait ~50ms so user can see
    key = cv2.waitKey(50) & 0xFF

    # Save every ~3 frames to avoid duplicates too quickly
    # Or just save every frame
    cv2.imwrite(os.path.join(user_folder, f"{user_name}_{count}.jpg"), frame)
    count += 1

    if count >= num_images:
        print("Captured all images.")
        break

    if key == ord('q'):
        print("User quit capturing.")
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------
# 3) Train a CNN on All Faces in dataset/
# ---------------------------
print("Training the model on all faces in 'dataset/'...")

# Basic parameters
img_size = (64, 64)
batch_size = 16
epochs = 10

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_data.num_classes

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("Starting training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)
print("Training complete. Saving model as 'face_rec_model.h5'")
model.save("face_rec_model.h5")

# Create a mapping from class index to name
class_indices = train_data.class_indices  # e.g. {'Mohammed': 0, 'Ali': 1, ...}
idx_to_name = {v: k for k, v in class_indices.items()}

# ---------------------------
# 4) Real-Time Recognition
# ---------------------------
print("Starting real-time recognition. Press 'q' to quit.")

model = load_model("face_rec_model.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB, resize, normalize
    img_resized = cv2.resize(frame, img_size)
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict
    preds = model.predict(img_resized, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)
    name = idx_to_name[class_id]

    # Display label
    cv2.putText(frame, f"{name} ({confidence*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Real-Time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")
