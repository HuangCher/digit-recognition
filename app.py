#install live server extension in vs code to run this code
#install flask w/ pip install flask 
#install numpy w/ pip install numpy
#install flask-cors w/ pip install flask-cors

from flask import Flask, request, jsonify
import numpy as np
import pickle
from cnn.cnn import *
from mlp.backend.mlp import *
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
    cnnPixelsArr = np.array(pixels)
    mlpPixelsArr = np.array(pixels)

    cnnPixelsArr = np.flip(cnnPixelsArr, axis=0)
    cnnPixelsArr = cnnPixelsArr[np.newaxis, :, :]

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
    cnnOutput = cnn.forward(cnnPixelsArr)
    predictionCNN = int(np.argmax(cnnOutput))

    #load MLP weights
    layerList = readLayersFromFile("MLP/backend/layers.txt")

    mlpOutput = forwardPropagate(layerList, mlpPixelsArr)
    predictionMLP = getResult(mlpOutput)

    return jsonify({'predictionCNN': predictionCNN, 'predictionMLP': predictionMLP})

if __name__ == '__main__':
    app.run(debug=True)
