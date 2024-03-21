from flask import Flask,render_template,request
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__,template_folder = 'template')


Model = tf.keras.models.load_model('/home/suraj/Desktop/jupyter/tomato classification_react_incomplete/models/1')
Class_names = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


# def read_file_as_image(data) -> np.ndarray:
#     # print(data)
#     with open(data, 'rb') as file:
#         image_data = file.read()

#     image = np.array(Image.open(BytesIO(image_data)))
#     return image

def predict_label(image_path):
    # img = image.load_img(image_path)
    # img1 = read_file_as_image(image_path)

    img1 = Image.open(image_path)
    image_rgb = img1.convert('RGB')
    img2 = image_rgb.resize((256,256))
    image_arr = tf.keras.preprocessing.image.img_to_array(img2)
    img_batch = np.expand_dims(image_arr, 0)
    predictions = Model.predict(img_batch)

    predicted_class = Class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.route('/',methods = ['GET','POST'])
def cv():
    return render_template('index.html')


@app.route('/predict',methods = ['GET','POST'])
def predict():

    if request.method == 'POST':
        img = request.files['my_image']
        img_path = 'static/'+img.filename
        img.save(img_path)

        p = predict_label(img_path)
    return render_template("index.html", prediction = p, img_path = img_path)
if __name__ == '__main__':
    app.run(debug = True)