from flask import Flask, render_template, request, jsonify, make_response
import requests
import sys
import os
from os.path import join, exists
import openai
from transformers import pipeline, Conversation
sys.path.append(join(os.getcwd(), 'src'))
from logger import logger
from software_recommender import Zero_Shot_Recosys


api_key = "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"
openai.api_key = api_key

# webapp
app = Flask(__name__, template_folder='web_interface')

reviews_path = "data/reviews.csv"
software_path = "data/softwares.csv"
query_params = {
"software_description" : "HR program for windows and mac for japanese humans",
"max_price" : 500,
"min_price" : 10
}
obj = Zero_Shot_Recosys(reviews_path, software_path)

@app.route("/")
def home():
    return render_template("home.html")
@app.route('/recommender')
def recommend():
    data = request.get_json()
    print(data)
    output = obj.recommender(query_params)
    out_data = output.to_json(orient="records")
    print(out_data)
    return jsonify({"output": out_data})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
