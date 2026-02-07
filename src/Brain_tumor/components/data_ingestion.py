import os
from dataclasses import dataclass
from tensorflow.keras.preprocessing.image import ImageDataGenerator

@dataclass
class DataIngestionConfig:
    training_dir = "archive/Training"
    testing_dir = "archive/Testing"

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def load_data(self):

        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2   # ✅ ONLY HERE
        )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_data = datagen.flow_from_directory(
            self.config.training_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="training",
            shuffle=True
        )

        val_data = datagen.flow_from_directory(   # ✅ SAME datagen
            self.config.training_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="validation",
            shuffle=True
        )

        test_data = test_datagen.flow_from_directory(
            self.config.testing_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            shuffle=False
        )

        print("All classes:", train_data.class_indices)
        print("Train batches:", len(train_data))
        print("Val batches:", len(val_data))

        return train_data.class_indices, train_data, val_data, test_data
