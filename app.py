#install live server extension in vs code to run this code
#install flask w/ pip install flask 
#install numpy w/ pip install numpy
#install flask-cors w/ pip install flask-cors

from flask import Flask, request, jsonify
import numpy as np
import pickle
from cnn.cnn import *
from MLP.backend.mlp import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# route to check if the server is running (ignore)
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
    
    #convert pixels to numpy array for CNN
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

    #predicts for cnn
    output = cnn.forward(pixelsArr)
    predictionCNN = int(np.argmax(output))

    #load MLP weights
    layerList = readLayersFromFile("MLP/backend/layers.txt")
    
    #flips the pixels array before sending it to the MLP model
    pixelsArr = np.flip(pixelsArr, axis=0)
    # dont change these (they are necessary for the model to read in the pixel data)
    # pixelsArr = pixelsArr * 255.0
    pixelsArr = 1.0 - pixelsArr
    pixelsArr = pixelsArr.flatten()
    pixelsArr = pixelsArr.reshape((784, 1))

    #predicts for mlp
    output = forwardPropagate(layerList, pixelsArr)
    predictionMLP = getResult(output)

    return jsonify({'predictionCNN': predictionCNN, 'predictionMLP': predictionMLP})

if __name__ == '__main__':
    app.run(debug=True)
