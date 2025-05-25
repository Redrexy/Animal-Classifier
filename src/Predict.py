# Load pre-trained model
import numpy as np
from PIL import Image
from keras.src.saving import load_model

# Load pre-trained model
model1 = load_model('animals_0_1.h5')
model2 = load_model('animals_0_255.h5')
class_names = ['Bird', 'Cat', 'Dog','Marine', 'Snake']

def predict(image_path):
    # Load image and convert to RGB (or "L" if grayscale)
    image = Image.open(image_path).convert("RGB")

    # Resize image to 128x128
    resized = image.resize((128, 128))

    # Convert to numpy array
    img_array = np.array(resized)

    # Reshape to match the input shape expected by the model (1, 128, 128, 3)
    img_array = img_array.reshape(1, 128, 128, 3)

    # Make predictions
    prediction = model1.predict(img_array)
    prediction2 = model2.predict(img_array)

    # Get predicted class and the confidence
    predicted_class1 = np.argmax(prediction)
    confidence1 = np.max(prediction)
    predicted_class2 = np.argmax(prediction2)
    confidence2 = np.max(prediction2)

    # Print prediction and confidence
    print('Prediction 1:', class_names[predicted_class1])
    print('Confidence 1:', confidence1)
    print('Prediction 2:', class_names[predicted_class2])
    print('Confidence 2:', confidence2)


predict('test/1bird.jpg')
predict('test/1cat.jpg')
predict('test/1dog.jpeg')
predict('test/1marine.jpg')
predict('test/1snake.webp')