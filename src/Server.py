import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from keras.src.saving import load_model
from flask_cors import CORS

app = Flask(__name__)  #App instance
CORS(app) # Other ports can connect

# Load pre-trained model
model1 = load_model('animals_0_1.h5')
model2 = load_model('animals_0_255.h5')
class_names = ['Bird', 'Cat', 'Dog','Marine', 'Snake']


# Test API
@app.route('/')
def get_test():
    return 'App is working!'


#Prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Get the uploaded image
    user_image = request.files['image']

    try:
        # Load image and convert to RGB
        image = Image.open(user_image).convert("RGB")

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

        # Return result as JSON
        return jsonify({
            'predicted_class1': class_names[predicted_class1],
            'confidence1': float(confidence1),
            'predicted_class2': class_names[predicted_class2],
            'confidence2': float(confidence2),
        })

    # Throw exception
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
