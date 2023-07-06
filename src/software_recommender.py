import os
import sys
from os.path import join

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(join(os.getcwd(), 'src'))
from logger import logger


class Zero_Shot_Recosys:
    SENTIMENT_LBLS = ["positive", "negative"]

    def ranking_algol(df):
        df['rank_score'] = (df['postive_score'] - df['negative_score']) * df['number_reviews']
        df = df.sort_values(by='rank_score', ascending=False)
        return df

    def price_ranking(max_price, min_price, max_license_price, min_license_price, max_maintenance_price,
                      min_maintenance_price, max_implementation_price, min_implementation_price, ranked_data):
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
            
        ranked_implementation_data = ranked_data[
            (ranked_data.Licensing_Fee >= min_implementation_price) & (ranked_data.Licensing_Fee <= max_implementation_price)]
        if len(ranked_implementation_data) >= 1:
            ranked_data = ranked_implementation_data
            logger.info(f"Size after filtering with maintenance fee: {len(ranked_implementation_data)}")
        else:
            msg.append(
                "The data available does not have the maintenance fee between the range you specified after license fee filtering")

        return ranked_data, msg

    @classmethod
    def rec_softwares(cls, model_name, software_data, software_description, max_price=np.inf, min_price=-1,
                      max_license=np.inf, min_license=-1, max_maintenance=np.inf,
                      min_maintenance=-1, max_implementation=np.inf, min_implementation=-1):
        if max_price is None:
            max_price=np.inf
        if min_price is None:
            min_price = -1
        if max_license is None:
            max_license=np.inf
        if min_license is None:
            min_license = -1
        if max_maintenance is None:
            max_maintenance = np.inf
        if min_maintenance is None:
            min_maintenance = -1
        if max_implementation is None:
            max_implementation = np.inf
        if min_implementation is None:
            min_implementation = -1

        software_data['software_description'] = software_description.lower()
        software_data['description'] = software_data['description'].str.lower()

        if model_name == "TfidfVectorizer":
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(software_data[['software_description', 'description']])
            cosine_sim = cosine_similarity(X)
        else:
            model = SentenceTransformer(model_name)
            X_1 = model.encode(software_data['software_description'])
            X_2 = model.encode(software_data['description'])
            cosine_sim = cosine_similarity(X_1, X_2)

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
                                                   float(min_license),float(max_maintenance), float(min_maintenance), float(max_implementation), float(min_implementation), ranked_data)

        if msg:
            for m in msg:
                logger.info(m)

        if len(price_ranked_data) > 2:
            return price_ranked_data
        else:
            return ranked_data

    def recommender(self, softwares, query_params) -> pd.DataFrame:
        softwares = softwares.copy()
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        output = self.rec_softwares(model_name=model_name, software_data=softwares, **query_params)
        output.to_csv(f"output/{model_name.replace('/', '_')}_output.csv")
        output = output.reset_index()
        return output.head(5)
