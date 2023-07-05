import os
import sys
from os.path import join
import json

import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.append(join(os.getcwd(), 'src'))
from software_recommender import Zero_Shot_Recosys

software_path = "data/softwares_with_score.csv"
softwares = pd.read_csv(software_path)
obj = Zero_Shot_Recosys()

app = Flask(__name__, template_folder='web_interface')

@app.route("/")
def home():
    return render_template("home.html")


@app.route('/recommender', methods=['GET'])
def recommend():
    query_params = {
        "software_description": request.args.get('software_description', ''),
        "max_price": request.args.get('max_price', ''),
        "min_price": request.args.get('min_price', ''),
        "max_license": request.args.get('max_license', ''),
        "min_license": request.args.get('min_license', ''),
        "max_maintenance": request.args.get('max_maintenance', ''),
        "min_maintenance": request.args.get('min_maintenance', '')
    }

    output = obj.recommender(softwares, query_params)
    output = output.to_json(orient='index')
    return jsonify({"output": output})


if __name__ == '__main__':
    app.run(debug=True)
