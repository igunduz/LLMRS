import pandas as pd
import ast
import numpy as np
import sys
from os.path import join
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

sys.path.append(join(os.getcwd(), 'src'))
from logger import logger

class PreProcessor:
    """
    This class reads reviews and software meta data, cleans data from unnecessaary information and saves them as 
    reveiws.csv and softwares.csv.
    These output files are used as input for software_data_processor for further processing.
    """
    def __init__(self, meta_data_path, reviews_data_path) -> None:
        self.meta_data = pd.read_csv(meta_data_path)
        logger.info(f"meta data is read with size {len(self.meta_data)}")
        self.review_data = pd.read_csv(reviews_data_path)
        logger.info(f"reviews data is read with size {len(self.review_data)}")

    @staticmethod
    def cleaning_price(data)-> pd.DataFrame:
        """ Formats the price field of the data, all values are qouted in $"""
        logger.info("formatting price field to be number only")
        data = data.copy()
        data['price'] = data.price.str.replace('$','')
        data['price'] = data.price.str[:5]
        data['price'] = data['price'].str.extract('([0-9]+.[0-9]*)')
        data['price'] = data['price'].str.strip()
        data['price'] = data['price'].str.replace(',','')
        data['price'] = data['price'].astype(float)
        data['price'] = data['price'] * 100
        data = data[data.price.notnull()]
        return data
    
    @staticmethod
    def assign_values(data):
        """ 
        Assigns Licensing_Fee, Implemention_cost and Maintenance_cost based on software categories
        """
        logger.info("Generating other fee columns")
        data['software_category'] = data['category'].apply(lambda x: ast.literal_eval(x)[1])
        Min_license_fee = data.groupby('software_category').agg(Min_Licensing_Fee=('price', np.min)).reset_index()
        data = data.merge(Min_license_fee, on='software_category', how='inner')
        data['Licensing_Fee'] = data['Min_Licensing_Fee'] * 0.8
        data.drop(columns=['Min_Licensing_Fee'],inplace=True)
        data['Implemention_cost'] = data.apply(lambda row: row['price']*0.5, axis=1)
        data['Maintenance_cost'] = data.apply(lambda row: row['price']*0.1, axis=1)
        return data
    
    @staticmethod
    def preprocess_text_fields(text: str) ->str:
        """
        Remove all punctuations and html tags from text
        """
        text = str(text)
        text = text.replace("<div>",'')
        text = text.replace("</div>",'')
        text = text.replace("< br/ >",'')
        text = text.replace("< br >",'')
        text = text.replace("< b >",'')
        text = text.replace("< /b >",'')
        text = text.replace("``",'')
        text = text.replace("< strong >",'')
        text = text.replace("< /strong >",'')
        text = text.replace("[", '')
        text = text.replace("]", '')
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        processed_text = ' '.join(lemmatized_tokens)
        html_form = re.compile(r'<.*?>')
        return html_form.sub('', processed_text)
        
        
    def main(self):
        software_data = self.meta_data.copy()
        review_data = self.review_data.copy()
        software_data = self.cleaning_price(software_data)
        software_data = self.assign_values(software_data)
        logger.info("Text fields cleaning")
        software_data['description'] = software_data['description'].apply(self.preprocess_text_fields)
        software_data['title'] = software_data['title'].apply(self.preprocess_text_fields)
        review_data['reviewText'] = review_data['reviewText'].apply(self.preprocess_text_fields)
        review_data['summary'] = review_data['summary'].apply(self.preprocess_text_fields)
        logger.info("Text fields cleaning complete")

        return review_data, software_data


if __name__ == "__main__":
    path_to_meta_data = "../external_data/filtered_metadata.csv"
    path_to_reviews_data = "../external_data/reviews_full.csv"
    obj = PreProcessor( path_to_meta_data, path_to_reviews_data )
    review_data, software_data = obj.main()
    review_data.to_csv("data/reviews.csv")
    logger.info("Reviews data cleaned and stored in /data")
    software_data.to_csv("data/softwares.csv")
    logger.info("Softwares data cleaned and stored in /data")