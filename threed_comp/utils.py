import pandas as pd
import os

def get_modelnet10_metadata():
    # read the metadata_modelnet10.csv file 
    meta_data_path = os.path.join(os.path.dirname(__file__), "metadata_modelnet10.csv")
    metadata = pd.read_csv(meta_data_path)
    return metadata
