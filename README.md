# LLMRS: Unlocking Potentials of LLM-Based Recommender Systems

## Downloading data
The dataset is in *.json* format [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/). After downloading *Software* reviews and metadata, we ran [this script](https://github.com/igunduz/sofRec/blob/main/src/notebooks/00_parse_and_clean_data.ipynb) to get data ready for preprocessing.

## Generating additional features
1. Licensing Fee is set to 80% of the minimum price in the software category. Llicensing fees could be similar in a particular software category and country.

2. Implementation Cost is set to 50% of the cost of the software.

3. Maintenance cost is assumed to be a monthly service so it was set to 1% of the price of the product.

## Setup environment
1. ``` pip install --upgrade pip ```
2. ``` python -m venv .llmrs ```
3. ``` source .llmrs/bin/activate ```
4. ``` conda deactivate ```
5. ``` pip install -r requirements.txt```

## Note: All monetary values are in USD($)

## To run recommendation
1. run ```python src/recommendation_api.py```

2. visit ``` 127.0.0.1:500 ```

    a. Enter Software description with price, license, maintenace and implementation costs in the respective boxes.

    b. When you clisk `Get Recommendation`, this would load pre-processed `data/softwares_with_scores.csv` and compute similarity with input software specification from user input.

    c. Output is then ranked with our ranking algorithm and parsed to the web interface




## The pipeline
The pipeline contains 3 steps as follows:
1. [Data preprocessing](https://github.com/igunduz/sofRec/blob/main/src/data_preprocessing.py)
2. [Process Software Data](https://github.com/igunduz/sofRec/blob/main/src/software_data_processor.py)
3. [Recommender](https://github.com/igunduz/sofRec/blob/main/src/notebooks/recommendation_api.py)

