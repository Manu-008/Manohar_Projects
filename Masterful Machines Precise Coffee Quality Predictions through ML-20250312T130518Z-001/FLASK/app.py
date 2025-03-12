import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route('/')
def hom():
    return render_template('index.html')

@app.route("/home",methods=["GET","POST"])
def home():
    if request.method=='POST':
        x = [x for x in request.form.values()]
        y = []
        for i in x:
            if type(i) == float:
                y.append(round(i))
            else:
                y.append(i)    

        x = pd.DataFrame(y).T 
        X = scaler.transform(x)
       
        predict=model.predict(X)
        return render_template('index.html',predict=predict)
    return render_template('index.html')
                           


if __name__ == "__main__":
    app.run(debug=True)