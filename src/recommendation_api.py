import os
import sys
from os.path import join

from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.append(join(os.getcwd(), 'src'))
from software_recommender import Zero_Shot_Recosys


app = Flask(__name__, template_folder='web_interface')

reviews_path = "data/reviews.csv"
software_path = "data/softwares.csv"
query_params = {
    "software_description": "HR program for windows and mac for japanese humans",
    "max_price": 500,
    "min_price": 10
}
obj = Zero_Shot_Recosys(reviews_path, software_path)


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('static', filename)
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

    output = obj.recommender(query_params)
    out_data = output.to_json(orient="index")
    return jsonify({"output": out_data})


if __name__ == '__main__':
    app.run(debug=True)
