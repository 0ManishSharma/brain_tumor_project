from dataclasses import dataclass
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,MaxPooling2D,Flatten,Input


@dataclass
class ModelTrainerConfig:
    artifacts_dir :str= os.path.join("artifacts",'model.h5')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.artifacts_dir),exist_ok=True)
    def build_model(self,num_class):
        
        model = Sequential()
        model.add(Input(shape=(224,224,3)))
        model.add(Conv2D(32,(3,3),activation="relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(64,(3,3),activation="relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(128,(3,3),activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(128,activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(num_class,activation="softmax"))

        return model
    
    def train(self,train_data,val_data,classes):
        num_class = len(classes)
        model = self.build_model(num_class)

        model.compile(
            optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ['accuracy']

        )
        model.fit(
            train_data,
            validation_data=val_data,
            batch_size=32,
            epochs=10,

        )
        model.save(self.config.artifacts_dir)
        print("Model Trained Successfully")
        