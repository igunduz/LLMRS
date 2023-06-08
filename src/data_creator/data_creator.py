import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

#print(scrapy.Request.url)

# URL of the website to scrape
BASE_URL = 'https://flippa.com/'
class Data_Creator:
    def __init__(self, config) -> None:
       self.config = config

    @staticmethod
    def get_variables(response_data):
        data = {}
        soup = BeautifulSoup(response_data.content, 'html.parser')
        # # Find specific elements on the page using their HTML tags, classes, or IDs
        # # For example, to extract all <a> tags, you can use:
        ask_price = soup.find('h5', class_='mb-0')
        location = soup.find('span', class_='pg-3 text-black')
        rating = soup.find('span', class_='text-green pl-1')
        all_links = soup.find_all('div', class_='pg-1 font-weight-bold')
        data['site_age'] = all_links[0].text
        data['monthly_profit'] = all_links[1].text
        data['profit_margin'] = all_links[2].text
        try:
            data['profit_multiple'] = all_links[3].text
        except Exception as e:
            data['profit_multiple'] = "NA"
        
        try:
            data['revenue_multiple'] = all_links[4].text
        except Exception as e:
            data['revenue_multiple'] = "NA"
        
        try:
            data['asking_price'] = ask_price.text
        except Exception as e:
            data['asking_price'] = "NA"
        try:
            data['location'] = location.text
        except Exception as e:
            data['location'] = "NA"
        try:
            data['rating'] = rating
        except Exception as e:
            data['rating'] = "NA"

        return data

    def main(self):      
        data = {'software_type':[],'site_age':[],'monthly_profit':[],'profit_margin':[],'profit_multiple':[],'revenue_multiple':[], 'price':[],'location':[], 'rating':[]}
        for software_type, listings in self.config.items():
            for list_id in listings:
                url = BASE_URL+ str(list_id)
                # Send a GET request to the website
                response = requests.get(url)
                # Check if the request was successful
                if response.status_code == 200:
                    out_data = self.get_variables(response)
                    data['software_type'].append(software_type)
                    data['site_age'].append(out_data['site_age'])
                    data['monthly_profit'].append(out_data['monthly_profit'])
                    data['profit_margin'].append(out_data['profit_margin'])
                    data['profit_multiple'].append(out_data['profit_multiple'])
                    data['revenue_multiple'].append(out_data['revenue_multiple'])
                    data['price'].append(out_data['asking_price'])
                    data['location'].append(out_data['location'])
                    data['rating'].append(out_data['rating'])
                else:
                    print('Failed to retrieve the webpage.')
        output_data = pd.DataFrame.from_dict(data)
        output_data.to_csv("src/flippa_data.csv")

if __name__ == "__main__":
    config_path = "src/configs/data_creator_config.json"
    with open(config_path) as pth:
        config = json.load(pth)
    #print(config)
    obj = Data_Creator(config)
    obj.main()