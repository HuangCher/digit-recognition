#install live server extension in vs code to run this code
#install flask w/ pip install flask 
#install numpy w/ pip install numpy
#install flask-cors w/ pip install flask-cors

from flask import Flask, request, jsonify
import numpy as np
import pickle
from cnn.cnn import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Flask server is running.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    pixels = data.get('pixels', None)

    #check if the user draw something
    if pixels is None:
        return jsonify({'error': 'No drawing provided.'}), 400
    
    #convert pixels to numpy array
    print(pixels)
    pixelsArr = np.array(pixels)
    pixelsArr = np.flip(pixelsArr, axis=0)
    pixelsArr = pixelsArr[np.newaxis, :, :]

    cnn = CNN()

    #load trained model weights
    try:
        with open('cnn/weights.pkl', 'rb') as f:
            weights = pickle.load(f)
            cnn.conv1_filters = weights['conv1_filters']
            cnn.conv1_bias = weights['conv1_bias']
            cnn.conv2_filters = weights['conv2_filters']
            cnn.conv2_bias = weights['conv2_bias']
            cnn.fc_weights = weights['fc_weights']
            cnn.fc_bias = weights['fc_bias']
        print("weights loaded.")
    except FileNotFoundError:
        print("weights file not found.")

    output = cnn.forward(pixelsArr)
    prediction = int(np.argmax(output))

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
