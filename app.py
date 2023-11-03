import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
classification_model=pickle.load(open('CKD_Model.pkl','rb'))
#scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=np.array(list(data.values())).reshape(1,-1)
    #new_data=np.array(list(data)).reshape(-1,1)
    output=classification_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    output = classification_model.predict(final_input)
    final_output = output[0]  # Store the prediction in the 'output' variable
    print(final_output)
    return render_template("home.html", prediction_text="Chronic Kidney Disease Classification {}".format(final_output))



if __name__=="__main__":
    app.run(debug=True)
   
     