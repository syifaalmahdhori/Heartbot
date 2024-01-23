from process import preparation, generate_response
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import load
import openai
import os 
# from dotenv import load_dotenv,find_dotenv

# __ = load_dotenv(find_dotenv()) #read local .env file
# openai.api_key=os.environ["OPENAI_API_KEY"]
# download nltk
preparation()

#Start Chatbot
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index3.html")
@app.route("/")
def chat():
    return render_template("index3.html")

model = None

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    
    return result

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk variabel input atau features (X) ke model
	input_age = 52
	input_sex  = 1
	input_cp = 0
	input_trestbps = 125
	input_chol = 212
	input_fbs = 0
	input_restecg = 1
	input_thalach = 168
	input_exang = 0
	input_oldpeak = 1
	input_slope = 2
	input_ca = 2
	input_thal = 3
	
	if request.method=='POST':
		# Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
		input_age = int(request.form['age'])
		input_sex  = int(request.form['sex'])
		input_cp = int(request.form['cp'])
		input_trestbps = int(request.form['trestbps'])
		input_chol  = int(request.form['chol'])
		input_fbs = int(request.form['fbs'])
		input_restecg = int(request.form['restecg'])
		input_thalach = int(request.form['thalach'])
		input_exang = int(request.form['exang'])
		input_oldpeak = float(request.form['oldpeak'])
		input_slope = float(request.form['slope'])
		input_ca = int(request.form['ca'])
		input_thal = int(request.form['thal'])
		# Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
		data_test = pd.DataFrame(data={
			"in_age" : [input_age],
			"in_sex"  : [input_sex],
			"in_cp" : [input_cp],
			"in_trestbps"  : [input_trestbps],
			"in_chol"  : [input_chol],
			"in_fbs"  : [input_fbs],
			"in_restecg"  : [input_restecg],
			"in_thalach"  : [input_thalach],
			"in_exang"  : [input_exang],
			"in_oldpeak"  : [input_oldpeak],
			"in_slope"  : [input_slope],
			"in_ca"  : [input_ca],
			"in_thal"  : [input_thal],
		})
        

		# Lakukan prediksi menggunakan model
		hasil_prediksi = model.predict(data_test[0:1])[0]

		# Menentukan tulisan hasil prediksi
		if hasil_prediksi == 0:
			tulisan_hasil_prediksi = "Tidak Memiliki Penyakit Jantung"
		else:
			tulisan_hasil_prediksi = "Memiliki Penyakit Jantung"

		# Return hasil prediksi dengan format JSON
		return jsonify({
			"tulisan_prediksi": tulisan_hasil_prediksi
		})


def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    # userText = request.args.get('msg')
    # completion = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": userText}
    # ]
    # )
    # result=completion.choices[0].message['content']
 
    return result



if __name__ == "__main__":
    model = load('model4_heartdisease_dt.model')
    app.run(debug=True)