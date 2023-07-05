import os
import sys
from os.path import join, exists

import pandas as pd
from transformers import pipeline
from zipfile import ZipFile

sys.path.append(join(os.getcwd(), 'src'))
from logger import logger


class Software_Data_Process:
    SENTIMENT_LBLS = ["positive", "negative"]

    def __init__(self, reviews_path, softwares_path):
        self.softwares = pd.read_csv(softwares_path)
        logger.info(f"softwares data loaded with size {len(self.softwares)}")
        # review_path exist open file else extract from zip
        if not exists(reviews_path):
            with ZipFile("data/reviews.zip", 'r') as zip:
                zip.extractall()

        reviews = pd.read_csv(reviews_path)
        reviews = reviews.merge(self.softwares, on='asin', how='inner')
        reviews = reviews[
            ['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'reviewerName', 'reviewText', 'summary',
             'vote']]
        self.reviews = reviews.dropna(subset='reviewText')
        logger.info(f"reviews data loaded with size {len(self.reviews)}")

    def prepare_software_data(self):
        logger.info("preparing software data for recommendation")
        reviews = self.reviews.copy()
        softwares = self.softwares.copy()
        scores_df = self.compute_sentiment_scores_for_all_asin(reviews)
        # add scores data to softwares
        softwares = softwares.merge(scores_df, on='asin', how='inner')
        softwares.dropna(subset="description", inplace=True)
        logger.info(f"software data of size {len(softwares)} is stored")
        return softwares

    @classmethod
    def compute_sentiment_scores_for_all_asin(cls, reviews):
        unique_ids = reviews.asin.unique().tolist()
        scores_dict = {
            "asin": [],
            "postive_score": [],
            "negative_score": [],
            "number_reviews": []
        }
        logger.info(f"Computing sentiment scores for {len(unique_ids)} softwares")
        for id in unique_ids:
            postive_score, negative_score, len_df = cls.compute_sentiment_scores_for_single_asin(
                reviews[reviews.asin == id])
            scores_dict['asin'].append(id)
            scores_dict['postive_score'].append(postive_score)
            scores_dict['negative_score'].append(negative_score)
            scores_dict['number_reviews'].append(len_df)

        scores_df = pd.DataFrame.from_dict(scores_dict)
        logger.info(f"All sentiment scores computed")
        return scores_df

    @classmethod
    def compute_sentiment_scores_for_single_asin(cls, df):
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli", device=0)
        postive_score = 0
        negative_score = 0
        for index, row in df.iterrows():
            input_text = row['reviewText']

            model_dict = classifier(input_text, cls.SENTIMENT_LBLS, multiclass=True)
            result_dict = dict(zip(model_dict.get('labels'), model_dict.get('scores')))
            postive_score += result_dict.get('positive')
            negative_score += result_dict.get('negative')
        nos_records = df.shape[0]
        return postive_score, negative_score, nos_records

    def main(self):
        software_path_to_file = "data/softwares_with_score.csv"
        if exists(software_path_to_file):
            logger.info("file already exist")
        else:
            logger.info("software data with scores does not exist, it is being created ...")
            softwares = self.prepare_software_data()
            logger.info("data creation completed")
            softwares.to_csv("data/softwares_with_score.csv")

if __name__  == "__main__":
    reviews_path = "data/reviews.csv"
    software_path = "data/softwares.csv"
    obj = Software_Data_Process(reviews_path, software_path)
    obj.main()