from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'>>'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info("exporting collection as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame = utils.get_collection_as_dataFrame(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)
            
            logging.info("Replace NA with NAN")

            #save data in feature store
            df.replace(to_replace="na",value=np.NAN,inplace=True)

            logging.info("Create feture store folder if not available")

            #Create feature store folder
            featur_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(featur_store_dir,exist_ok=True)

            logging.info("save df in fature store folder")
            #save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)

            logging.info("split dataset into train and test")
            #split dataset into train and test set
            train_df,test_df=train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)

            logging.info("create dataset folder if not available")
            #create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("save train and test df in feature store folder")
            #save df to feature store folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            logging.info("preparing Artifact")
            #prepare artifact
            data_ingestion_artifact= artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact


        except Exception as e:
            raise SensorException(error_message=e,error_detail=sys)
