import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
from tensorflow.python.keras.callbacks import EarlyStopping


directory_data = 'classes'

# Training data
train_data = image_dataset_from_directory(
    directory_data,
    labels='inferred', # Assigns labels based on folder names
    label_mode='int', # Classes represented as integers (0, 1, 2)
    batch_size=32, # Each batch contains 32 images
    image_size=(128, 128), # Resize image
    shuffle=True, # Shuffle the data
    seed=123, # Reproducibility
    validation_split=0.2, # 20% is val
    subset="training", # Training data
)

# Validation data
val_data = image_dataset_from_directory(
    directory_data,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=123,
    validation_split=0.2,
    subset='validation',
)

class_names = train_data.class_names
def predict(image_path):
    # Load image and convert to RGB (or "L" if grayscale)
    image = Image.open(image_path).convert("RGB")

    # Resize image to 128x128
    resized = image.resize((128, 128))

    # Convert to numpy array
    img_array = np.array(resized)

    # Reshape to match the input shape expected by the model (1, 128, 128, 3)
    img_array = img_array.reshape(1, 128, 128, 3)

    # Make prediction
    prediction = model.predict(img_array)

    # Get predicted class and the confidence
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print('Prediction:', class_names[predicted_class])
    print('Confidence:', confidence)

# EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors validation loss
    patience=15,  # Will stop after no improvement in a set number of epochs
    restore_best_weights=True,  # Restore the best epoch
)

model = Sequential([
    # Layer 1 for pixel values [0, 1]: Rescaling to normalize pixel values to [0, 1]
    Rescaling(1./255, input_shape=(128, 128, 3)),  # Divides each pixel by 255 to normalize
    Conv2D(32, kernel_size=(3, 3), activation='relu'),

    # # Layer 1 for pixel values [0, 255]: Convolutional layer with 32 filters of size 3x3 and ReLU activation
    # Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),

    # Layer 2: Max pooling layer with pool size of (2, 2)
    MaxPooling2D((2, 2)),

    # Layer 3: Convolutional layer with 64 filters of size 3x3
    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    # Layer 4: Max pooling layer with pool size of (2, 2)
    MaxPooling2D((2, 2)),

    # Layer 5: Convolutional layer with 128 filters of size 3x3
    Conv2D(128, kernel_size=(3, 3), activation='relu'),

    # Layer 6: Max pooling layer with pool size of (2, 2)
    MaxPooling2D((2, 2)),

    # Layer 7: Dropout layer with a rate of 0.25 (25% of the neurons will be dropped)
    Dropout(0.25),

    # Layer 8: Flatten the 3D output into a 1D vector
    Flatten(),

    # Layer 9: Fully connected (Dense) layer with 128 neurons and ReLU activation
    Dense(128, activation='relu'),

    # Layer 10: Dropout layer with a rate of 0.25 (25% of the neurons will be dropped)
    Dropout(0.25),

    # Layer 11: Output layer with 5 neurons (for the 5 classes) and softmax activation for multi-class classification
    Dense(5, activation='softmax'),
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Summary of the model
# model.summary()

# Train the model
model.fit(train_data, validation_data=val_data, epochs=100, callbacks=[early_stopping])

# # Save the trained model to a file (HDF5 format)
# model.save('animals_0_1.h5')

predict('test/1bird.jpg')
predict('test/1cat.jpg')
predict('test/1dog.jpeg')
predict('test/1marine.jpg')
predict('test/1snake.webp')