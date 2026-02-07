from dataclasses import dataclass
import os

@dataclass
class DataValidationConfig:
    train_dir = "archive/Training"
    
class DataValidation:
    def __init__(self):
        self.config = DataValidationConfig()
    
    def validate(self):
        if not os.path.exists(self.config.train_dir):
            raise FileNotFoundError("Dataset Folder is not found")
        classes = os.listdir(self.config.train_dir)

        if len(classes) == 0:
            raise ValueError("No class is found")
        for cls in classes:

            class_path = os.path.join(self.config.train_dir,cls)

            if not os.path.isdir(class_path):
                raise ValueError(f'No {cls} is found')
            
            images = os.listdir(class_path)
            if len(images) == 0:
                raise ValueError(f"No images in this {cls}")
            
        print("Data Validation Successfully")

        return True

