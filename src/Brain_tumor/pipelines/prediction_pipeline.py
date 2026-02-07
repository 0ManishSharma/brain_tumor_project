import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class Prediction:
    def __init__(self,model_path,classes):
        self.model = load_model(model_path)
        self.classes = classes
        self.index_to_class = {v: k for k, v in self.classes.items()}
    def predict(self,image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = self.model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        confidence=np.max(prediction[0])

        predicted_class = self.index_to_class[predicted_index]

        return predicted_class, float(confidence)