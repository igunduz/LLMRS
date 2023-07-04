import pandas as pd
import ast
import numpy as np
import sys
from os.path import join
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

sys.path.append(join(os.getcwd(), 'src'))
from logger import logger

class PreProcessor:
    def __init__(self, meta_data_path, reviews_data_path) -> None:
        self.meta_data = pd.read_csv(meta_data_path)
        self.review_data = pd.read_csv(reviews_data_path)
        #self.data = review_data.merge(meta_data, on="asin", how="inner")

    @staticmethod
    def cleaning_price(data)-> pd.DataFrame:
        data = data.copy()
        #data = data[data.price.notnull()]
        data['price'] = data.price.str.replace('$','')
        data['price'] = data.price.str[:5]
        data['price'] = data['price'].str.extract('([0-9]+.[0-9]*)')
        data['price'] = data['price'].str.strip()
        data['price'] = data['price'].str.replace(',','')
        data['price'] = data['price'].astype(float)
        data = data[data.price.notnull()]
        return data
    
    @staticmethod
    def assign_values(data):
        data['software_category'] = data['category'].apply(lambda x: ast.literal_eval(x)[1])
        Min_license_fee = data.groupby('software_category').agg(Min_Licensing_Fee=('price', np.min)).reset_index()
        data = data.merge(Min_license_fee, on='software_category', how='inner')
        data['Licensing_Fee'] = data['Min_Licensing_Fee'] * 0.8
        data.drop(columns=['Min_Licensing_Fee'],inplace=True)
        data['Implemention_cost'] = data.apply(lambda row: row['price']*0.5, axis=1)
        data['Maintenance_cost'] = data.apply(lambda row: row['price']*0.1, axis=1)
        return data
    
    @staticmethod
    def preprocess_text_fields(text):
        # Tokenize the text
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
        tokens = word_tokenize(text.lower())
        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # Join the tokens back into a string
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text
    
    def main(self):
        software_data = self.meta_data.copy()
        review_data = self.review_data.copy()
        software_data = self.cleaning_price(software_data)
        software_data = self.assign_values(software_data)
        software_data['description'] = software_data['description'].apply(self.preprocess_text_fields)
        software_data['title'] = software_data['title'].apply(self.preprocess_text_fields)

        review_data['reviewText'] = review_data['reviewText'].apply(self.preprocess_text_fields)
        review_data['summary'] = review_data['summary'].apply(self.preprocess_text_fields)

        review_data.to_csv("../data/reviews.csv")
        software_data.to_csv("../data/softwares.csv")
        logger.info("review_data is cleaned and saved in data/reviews.csv and softwares_data is saved in data/softwares.csv")

# if __name__ == "__main__":
#     obj = PreProcessor("../external_data/filtered_metadata.csv", "../external_data/reviews_full.csv")
#     obj.main()