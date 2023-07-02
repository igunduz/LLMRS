# sofRec: Recommender system for software solutions

## Downloading data
The dataset is available as *.json* format [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/). After downloading *Software* reviews and metadata, we ran [this script](https://github.com/igunduz/sofRec/blob/main/src/notebooks/00_parse_and_clean_data.ipynb) to get data ready for preprocessing.
## Pre-Processing data
1. Licensing Fee is set to 80% of minimum price in the software category. The idea is that licensing fee could be a similar in a particular sostware category and country.

2. Implementation Cost is set to 50% of cost of software.

3. Maintenace cost is assumed to be a monthly service so was set to 1% of cost of product.

To run the data preproceesor run:
```python src/data_preprocessing.py ```

## The pipeline
The pipeline contains 5 steps as follows:

1. [Data preprocessing](https://github.com/igunduz/sofRec/blob/main/src/notebooks/01_data_preprocessing.ipynb)
2. [basic analysis](https://github.com/igunduz/sofRec/blob/main/src/notebooks/02_basic_analysis.ipynb)
3. [LLM encoding](https://github.com/igunduz/sofRec/blob/main/src/notebooks/03_LLM_encoding.ipynb)
4. [Data Labeling Transformer](https://github.com/igunduz/sofRec/blob/main/src/notebooks/04_data_Labeling_Transformer.ipynb)
5. [Cosine similarity](https://github.com/igunduz/sofRec/blob/main/src/notebooks/05_cosine_similarity.ipynb)
