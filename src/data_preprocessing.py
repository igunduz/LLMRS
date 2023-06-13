import pandas as pd
import ast
import numpy as np
import sys
from os.path import join
import os

sys.path.append(join(os.getcwd(), 'src'))
from logger import logger

class PreProcessor:
    def __init__(self, data_path) -> None:
        self.data = pd.read_csv(data_path)

    @staticmethod
    def data_cleaning(data)-> pd.DataFrame:
        logger.info(f"Data is read with size {len(data)}")
        #data = data[data.price.notnull()]
        data['price'] = data.price.str.replace('$','')[:5] # get only value
        data['price'] = data['price'].astype(float)
        data = data[data.price.notnull()]
        logger.info(f"Price filtered by NA , size after filtering is {len(data)}")
        data['software_category'] = data['category'].apply(lambda x: ast.literal_eval(x)[1])
        return data
    
    @staticmethod
    def assign_values(data):
        Min_license_fee = data.groupby('software_category').agg(Min_Licensing_Fee=('price', np.min)).reset_index()
        data = data.merge(Min_license_fee, on='software_category', how='inner')
        data['Licensing_Fee'] = data['Min_Licensing_Fee'] * 0.8
        data.drop(columns=['Min_Licensing_Fee'],inplace=True)
        data['Implemention_cost'] = data.apply(lambda row: row['price']*0.5, axis=1)
        data['Maintenance_cost'] = data.apply(lambda row: row['price']*0.1, axis=1)
        return data
    
    def main(self):
        data = self.data.copy()
        cleaned_data = self.data_cleaning(data)
        meta_data = self.assign_values(cleaned_data)
        meta_data.to_csv("data/cleaned_meta_data.csv")
        logger.info("Data cleaned and saved in data/cleaned_metadata.csv")

if __name__ == "__main__":
    obj = PreProcessor("data/metadata_subset.csv")
    obj.main()