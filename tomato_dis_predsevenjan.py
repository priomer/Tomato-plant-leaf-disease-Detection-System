# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:38:59 2021

@author: ocn
"""

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(200, 200))
    #img.show()

    enhancer_1 = ImageEnhance.Contrast(img)
    imgs_con = enhancer_1.enhance(0.8)
    plt.figure(figsize=(10,10))
    plt.imshow(imgs_con)

    enhancer_2 = ImageEnhance.Color(imgs_con)
    imgs_col = enhancer_2.enhance(0.9)
    plt.figure(figsize=(10,10))
    plt.imshow(imgs_col)
 
    enhancer_3 = ImageEnhance.Brightness(imgs_col)
    imgs_bri = enhancer_3.enhance(0.8)
    plt.figure(figsize=(10,10))
    plt.imshow(imgs_bri)
    img_tensor= image.img_to_array(imgs_bri)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    
    return img_tensor
    
model = load_model("model_tomato_plant_disease4.h5")
img_path = 'E:/AVRN_Report/Plant_Diseases_Dataset/test/septoria/septoria3.jpg'
check_image = load_image(img_path)
prediction = model.predict(check_image)
print(prediction)

prediction =np.argmax(prediction, axis=1)
if prediction==0:
    prediction="Bacterial_spot"
elif prediction==1:
    prediction="Early_blight"
elif prediction==2:
    prediction="Late_blight"
elif prediction==3:
    prediction="Leaf_Mold"
elif prediction==4:
    prediction="Septoria_leaf_spot"
elif prediction==5:
    prediction="Spider_mites Two-spotted_spider_mite"
elif prediction==6:
    prediction="Target_Spot"
elif prediction==7:
    prediction="Tomato_Yellow_Leaf_Curl_Virus"
elif prediction==8:
    prediction="Tomato_mosaic_virus"
else:
    prediction="Healthy"

print(prediction)    
    
    

