import os
import sys
from os.path import join, exists

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

sys.path.append(join(os.getcwd(), 'src'))
from logger import logger


class Zero_Shot_Recosys:
    SENTIMENT_LBLS = ["positive", "negative"]

    def __init__(self, reviews_path, softwares_path):
        reviews = pd.read_csv(reviews_path)
        self.softwares = pd.read_csv(softwares_path)
        logger.info(f"softwares data loaded with size {len(self.softwares)}")
        reviews = reviews.merge(self.softwares, on='asin', how='inner')
        reviews = reviews[
            ['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'reviewerName', 'reviewText', 'summary',
             'vote']]
        self.reviews = reviews.dropna(subset='reviewText')
        logger.info(f"reviews data loaded with size {len(self.reviews)}")

    @classmethod
    def compute_sentiment_scores_for_single_asin(cls, df):
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli", device=0)
        postive_score = 0
        negative_score = 0
        for index, row in df.iterrows():
            input_text = row['reviewText']

            model_dict = classifier(input_text, cls.SENTIMENT_LBLS, multi_label=True)
            result_dict = dict(zip(model_dict.get('labels'), model_dict.get('scores')))
            postive_score += result_dict.get('positive')
            negative_score += result_dict.get('negative')
        nos_records = df.shape[0]
        return postive_score, negative_score, nos_records

    @classmethod
    def compute_sentiment_scores_for_all_asin(cls, reviews):
        unique_ids = reviews.asin.unique().tolist()
        scores_dict = {
            "asin": [],
            "postive_score": [],
            "negative_score": [],
            "number_reviews": []
        }
        logger.info(f"Number of softwares available is {len(unique_ids)}")
        for id in unique_ids:
            postive_score, negative_score, len_df = cls.compute_sentiment_scores_for_single_asin(
                reviews[reviews.asin == id])
            scores_dict['asin'].append(id)
            scores_dict['postive_score'].append(postive_score)
            scores_dict['negative_score'].append(negative_score)
            scores_dict['number_reviews'].append(len_df)

        scores_df = pd.DataFrame.from_dict(scores_dict)
        return scores_df

    def ranking_algol(df):
        df['rank_score'] = (df['postive_score'] - df['negative_score']) * df['number_reviews']
        df = df.sort_values(by='rank_score', ascending=False)
        return df

    def price_ranking(max_price, min_price, max_license_price, min_license_price, max_maintenance_price,
                      min_maintenance_price, ranked_data):
        msg = []
        ranked_price_data = ranked_data[(ranked_data.price >= min_price) & (ranked_data.price <= max_price)]
        if len(ranked_price_data) >= 1:
            ranked_data = ranked_price_data
            logger.info(f"Size after filtering with price: {len(ranked_price_data)}")
        else:
            msg.append("The data available does not have price between the range you specified")

        ranked_license_data = ranked_data[
            (ranked_data.Licensing_Fee >= min_license_price) & (ranked_data.Licensing_Fee <= max_license_price)]
        if len(ranked_license_data) >= 1:
            ranked_data = ranked_license_data
            logger.info(f"Size after filtering with license fee: {len(ranked_license_data)}")
        else:
            msg.append(
                "The data available does not have license fee between the range you specified after the price filtering.")

        ranked_maintenance_data = ranked_data[
            (ranked_data.Licensing_Fee >= min_maintenance_price) & (ranked_data.Licensing_Fee <= max_maintenance_price)]
        if len(ranked_maintenance_data) >= 1:
            ranked_data = ranked_maintenance_data
            logger.info(f"Size after filtering with maintenance fee: {len(ranked_maintenance_data)}")
        else:
            msg.append(
                "The data available does not have the maintenance fee between the range you specified after license fee filtering")

        return ranked_data, msg

    @classmethod
    def rec_softwares(cls, model_name, software_data, software_description, max_price=np.inf, min_price=-1,
                      max_license=np.inf, min_license=-1, max_maintenance=np.inf,
                      min_maintenance=-1):
        software_data['software_description'] = software_description

        if model_name == "TfidfVectorizer":
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(software_data[['software_description', 'description']])
            cosine_sim = cosine_similarity(X)
        else:
            try:
                model = SentenceTransformer(model_name)
                X_1 = model.encode(software_data['software_description'])
                X_2 = model.encode(software_data['description'])
                cosine_sim = cosine_similarity(X_1, X_2)
            except Exception as e:
                raise f"{model_name} not a transformer model or TfidVectorizer"

        target_item_index = 0
        scores = list(enumerate(cosine_sim[target_item_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_n = 10
        top_recommendations = scores[0:top_n + 1]
        index_list = []
        for ind, score in top_recommendations:
            index_list.append(ind)
        ranked_data = cls.ranking_algol(software_data.iloc[index_list, :])
        price_ranked_data, msg = cls.price_ranking(float(max_price), float(min_price), float(max_license),
                                                   float(min_license),
                                                   float(max_maintenance), float(min_maintenance), ranked_data)

        if msg:
            for m in msg:
                logger.info(m)

        if len(price_ranked_data) > 2:
            return price_ranked_data
        else:
            return ranked_data

    def prepare_software_data(self):
        reviews = self.reviews.copy()
        softwares = self.softwares.copy()
        scores_df = self.compute_sentiment_scores_for_all_asin(reviews)
        # add scores data to softwares
        softwares = softwares.merge(scores_df, on='asin', how='inner')

        return softwares

    def recommender(self, query_params) -> pd.DataFrame:
        software_path_to_file = "data/softwares_only__.csv"
        if exists(software_path_to_file):
            logger.info("file exist and has been read")
            softwares = pd.read_csv(software_path_to_file)
        else:
            softwares = self.prepare_software_data()
            softwares.to_csv("data/softwares_with_score.csv")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        output = self.rec_softwares(model_name=model_name, software_data=softwares, **query_params)
        output.to_csv(f"output/{model_name.replace('/', '_')}_output.csv")
        return output.head(2)
