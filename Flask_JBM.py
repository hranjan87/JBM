from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from Flash2.load_model import *
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from werkzeug.utils import secure_filename


app = Flask(__name__)
global model, graph
model, graph = init()


def model_predict(img_path):
    img_width, img_height = 150, 150
    model_path = '/Users/HimanshuRanjan/MachineLearning/JBM/models/model.h5'
    model_weight_path = '/Users/HimanshuRanjan/MachineLearning/JBM/models/weights.h5'
    model = load_model(model_path)
    model.load_weights(model_weight_path)
    x = load_img(imagepath, target_size = (img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis =0)
    array = model.predict(x)
    category = array[0]
    return np.argmax(category)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)

        # Make prediction
        preds = model_predict(img_path)
        return preds
    return None


if __name__ == '__main__':
    app.run(port=5002, debug=True)