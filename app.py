from src.Brain_tumor.components.data_ingestion import DataIngestion
from src.Brain_tumor.components.data_validation import DataValidation
from src.Brain_tumor.components.model_tranier import ModelTrainer

if __name__ == "__main__":

    data_ingestion = DataIngestion()
    classes,train_data,val_data,test_data = data_ingestion.load_data()
    print(classes)
    # data_validation = DataValidation()
    # data_validation.validate()
    # print("len of train_data:",len(train_data))
    # print("len of the val_data",len(val_data))
    # model_train = ModelTrainer()
    # model_train.train(train_data,val_data,classes)
    