from sensor.entity import artifact_entity
from sensor.entity import config_entity
from sensor.exception import SensorException
from sensor.logger import logging
import sys,os
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score



class ModelTrainer:
    
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def fine_tune(self):
        #Grid search CV
        try:
            pass
        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self,x,y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    
    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.tranformation_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.tranformation_test_path)

            logging.info("splitting input and target feature from both train and test array")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info("train model")
            model = self.train_model(x=x_train,y=y_train)

            logging.info("calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train,y_pred=yhat_train)
            
            logging.info("Calculating f1 tets score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test,y_pred=yhat_test)

            logging.info(f" model train score : {f1_train_score} and model test acore : {f1_test_score}")
            #check for overfitting or underfitting or expected score
            logging.info("Checking if mode is underfitted or not")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                    expected accuracy: {self.model_trainer_config.expected_score} : Model actual score: {f1_test_score}")

            logging.info("checking if model is overfitted or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff > self.model_trainer_config.overfitting_thres:
                raise Exception(f"Model is overfitting as Train and test score diff,{diff}, is more than model thresold : {self.ModelTrainerConfig.overfitting_thres}")


            #save the trained model
            logging.info("saving model object")
            utils.save_obects(file_path= self.model_trainer_config.model_path , obj=model)

            #prepare artifact
            logging.info("preparing artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path 
            , f1_train_score=f1_test_score
            , f1_test_score=f1_test_score)


            logging.info(f"model trainer artifact : {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception  as e:
            raise SensorException(e, sys)

        