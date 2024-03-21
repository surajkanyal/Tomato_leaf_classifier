# from flask import Flask,render_template,request
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from keras.preprocessing import image

# app = Flask(__name__,template_folder = 'template')


# Model = tf.keras.models.load_model('/home/suraj/Desktop/jupyter/tomato classification_react_incomplete/models/1')
# Class_names = ['Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___healthy']


# # def read_file_as_image(data) -> np.ndarray:
# #     # print(data)
# #     with open(data, 'rb') as file:
# #         image_data = file.read()

# #     image = np.array(Image.open(BytesIO(image_data)))
# #     return image

# def predict_label(image_path):
#     # img = image.load_img(image_path)
#     # img1 = read_file_as_image(image_path)

#     img1 = Image.open(image_path)
#     image_rgb = img1.convert('RGB')
#     img2 = image_rgb.resize((256,256))
#     image_arr = tf.keras.preprocessing.image.img_to_array(img2)
#     img_batch = np.expand_dims(image_arr, 0)
#     predictions = Model.predict(img_batch)

#     predicted_class = Class_names[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }


# @app.route('/',methods = ['GET','POST'])
# def cv():
#     return render_template('index.html')


# @app.route('/predict',methods = ['GET','POST'])
# def predict():

#     if request.method == 'POST':
#         img = request.files['my_image']
#         img_path = 'static/'+img.filename
#         img.save(img_path)

#         p = predict_label(img_path)
        
#     return render_template("index.html", prediction = p, img_path = img_path)
# if __name__ == '__main__':
#     app.run(debug = True)
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

def info(about):
    if about == Class_names[0]:
        return('''Tomato bacterial spot is a damaging plant disease caused by Xanthomonas campestris pv. vesicatoria.It affects tomatoes, peppers, and eggplants./ Symptoms include leaf lesions, fruit blemishes, and stem damage. Leaves develop small water-soaked lesions that turn brown or black with yellow halos, often leading to defoliation. Circular, raised lesions appear on fruit, which can crack as they mature, and stems may also display lesions.

Preventing and managing this disease involves:/

Resistant Varieties: Choose resistant tomato varieties./

Crop Rotation: Avoid planting susceptible crops in the same spot year after year./

Sanitation: Remove and destroy infected plant material, and clean tools./

Watering: Water at the base to prevent splashing.''')
    elif about == Class_names[1]:
        return('''Tomato early blight is a common fungal disease caused by the pathogen Alternaria solani. It affects tomato plants and can lead to significant damage to both leaves and fruit. Early blight is typically characterized by the appearance of small, dark lesions on the lower leaves of the tomato plant. These lesions often have a concentric ring pattern, giving them a target-like appearance. As the disease progresses, the affected leaves can turn yellow, wither, and eventually die. In severe cases, early blight can also affect the fruit, causing dark, sunken lesions on the tomato, which can render it unmarketable.
/
Preventing and managing early blight involves several strategies:/

Crop Rotation: Avoid planting tomatoes in the same location for consecutive years to reduce the buildup of the pathogen in the soil./

Sanitation: Remove and destroy infected plant material, including leaves and fruit, to prevent the spread of spores.
/
Proper Spacing: Plant tomatoes with adequate spacing to improve air circulation and reduce humidity around the plants.
/
Mulching: Use mulch to prevent soil splashing onto lower leaves, as the pathogen can be soil-borne.''')
    elif about == Class_names[2]:
        return('''Tomato late blight, caused by the pathogenic oomycete organism Phytophthora infestans, is a devastating fungal-like disease that primarily affects tomato plants and can lead to significant crop losses. This disease is infamous for its role in the Irish Potato Famine in the 19th century and continues to pose a serious threat to tomato and potato crops worldwide.
/
Symptoms:
/
Foliage: Early symptoms appear as dark, water-soaked lesions on the leaves, often surrounded by a pale yellow halo. These lesions can quickly enlarge and lead to the wilting and death of the plant's foliage.
/
Fruit: Late blight can also affect tomato fruit, causing dark, firm lesions that render the fruit inedible.
/
Prevention and Management:
/
Resistant Varieties: Plant late blight-resistant tomato varieties when available.
/
Crop Rotation: Avoid planting tomatoes or other susceptible crops in the same location for consecutive seasons.''')
    elif about == Class_names[3]:
        return('''Tomato leaf mold is a fungal disease caused by the pathogen Pseudocercospora fuligena (formerly known as Passalora fulva). This disease primarily affects tomato plants and can lead to significant reductions in fruit yield and quality. Leaf mold is particularly troublesome in areas with high humidity and moderate temperatures, as these conditions promote its development.
/
Symptoms of tomato leaf mold typically appear on the lower leaves of the plant and include:
/
Yellow to pale green patches: These develop on the upper leaf surface.
/Olive-green to brownish-gray fuzzy growth: This appears on the lower leaf surface.
/Preventing and managing tomato leaf mold involves several key strategies:
/
Proper spacing: Plant tomatoes at recommended spacing to allow for good air circulation and reduce humidity around the leaves.
/
Pruning: Regularly prune the lower leaves of the plant to improve airflow and reduce the chances of fungal spore contact.''')
    elif about == Class_names[4]:
        return('''Tomato Septoria leaf spot, caused by the fungus Septoria lycopersici, is a common and destructive disease affecting tomato plants. It primarily targets the leaves but can also affect stems and fruit. This disease is characterized by small, circular lesions with dark centers and light gray or tan margins on the lower leaves of tomato plants. /
               Prevention and control measures:/

Resistant Varieties: Choose tomato varieties that are bred for resistance to Septoria leaf spot. These varieties are less susceptible to infection.
/
Crop Rotation: Avoid planting tomatoes and related crops in the same location for consecutive growing seasons to reduce the buildup of fungal spores in the soil.
/
Pruning and Thinning: Promote good air circulation by pruning and thinning tomato plants to reduce humidity and minimize conditions conducive to fungal growth.''')
    elif about == Class_names[5]:
        return('''The Two-Spotted Spider Mite (Tetranychus urticae) is a tiny arachnid pest that infests various plants, including tomatoes. These spider mites are named for the two distinctive dark spots they have on their bodies. They are a common pest in many regions and can cause significant damage to tomato plants./
              Preventing and Managing Two-Spotted Spider Mites:/

Regular Inspection: Routinely inspect your tomato plants for early signs of infestation, including yellowing leaves and fine webbing.
/Pruning: Prune and remove heavily infested plant parts to reduce the mite population and prevent further spread.
/Isolation: Isolate new plants for a few weeks to ensure they are not carrying spider mites before introducing them to your garden. ''')
    elif about == Class_names[6]:
        return('''Tomato target spot, also known as early blight, is a common and destructive fungal disease that affects tomato plants (Solanum lycopersicum). It is caused by the fungus Alternaria solani. This disease primarily impacts the leaves and fruits of tomato plants and can significantly reduce crop yields if left untreated.

/Prevention and Management:
/
Crop Rotation: Avoid planting tomatoes or related crops in the same location year after year to reduce fungal spore buildup in the soil.
/
Resistant Varieties: Choose tomato varieties with resistance to target spot.
/
Sanitation:
/
Remove and destroy infected plant material.
Keep the garden clean, and remove debris to reduce overwintering fungal spores.              ''')
    elif about == Class_names[7]:
        return('''omato Yellow Leaf Curl Virus (TYLCV) is a highly destructive viral disease that affects tomato plants and other crops, causing significant damage to tomato production worldwide. TYLCV is primarily transmitted by the whitefly, an insect vector./
               Prevention for TYLCV are:/
               Whitefly Control: Implement rigorous whitefly control measures, including insecticides and physical barriers like row covers, to reduce vector populations.
/
Resistant Tomato Varieties: Plant TYLCV-resistant tomato varieties, which are bred to withstand the virus.
/
Sanitation: Remove and destroy infected plants promptly to prevent virus spread.
               ''')
    elif about == Class_names[8]:
        return('''Tomato mosaic virus (ToMV) is a highly infectious and damaging plant virus that primarily affects tomatoes but can also impact other plants in the Solanaceae family, such as peppers and eggplants. It belongs to the Tobamovirus genus and is known for causing mosaic-like patterns of light and dark green on infected tomato leaves, along with other symptoms. /
               Preventing and Managing Tomato Mosaic Virus:
/
Plant Resistant Varieties: Choose tomato varieties with resistance to ToMV.
/Sanitation: Maintain good hygiene practices, including hand washing and tool disinfection, to prevent the virus's spread.
/Control Aphids: Aphids can transmit ToMV, so implement measures to control aphid populations.''')
    else :
        return('''The given tomato is healthy.''')
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
        data = info(p['class'])
    return render_template("index.html", prediction = p, img_path = img_path,data = data)
if __name__ == '__main__':
    app.run(debug = True)


    # things to do card shape
    # insert data in info function
    # set home as active
    # create about html
    # Beautify the page