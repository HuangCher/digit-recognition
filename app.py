#install live server extension in vs code to run this code
#install flask w/ pip install flask 
#install numpy w/ pip install numpy
#install flask-cors w/ pip install flask-cors

from flask import Flask, request, jsonify
import numpy as np
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
    pixels_array = np.array(pixels)

    #placeholder
    prediction = 'test Prediction'

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
