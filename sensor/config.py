import pymongo
import pandas as pd
import json
from dataclasses import dataclass
import os
#provide the mongodb localhost url to connect python to mongodb

@dataclass
class Environmentvariable:
    mongo_db_url:str=os.getenv("MONGO_DB_URL")

env_var=Environmentvariable()

mongo_client=pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN ="class"

