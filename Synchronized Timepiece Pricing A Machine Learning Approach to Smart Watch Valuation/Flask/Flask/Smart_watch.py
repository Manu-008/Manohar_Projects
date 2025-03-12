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
        x = [int(x) for x in request.form.values()]
        
        x = np.array(x)
        x= x.reshape(-1,1)
 
        x = pd.DataFrame(x).T 
        X = scaler.transform(x)

        #labels = ['Brand', 'Operating System', 'Connectivity', 'Display Type','Display Size (inches)','Water Resistance (meters)','Battery Life (days)','GPS','NFC','Pixels']
       
        #X= pd.DataFrame(x,columns = labels)   
        #X = x.reshape(-1,1)
       
        predict=model.predict(X)
        return render_template('index.html',predict=predict)
    return render_template('index.html')
                           


if __name__ == "__main__":
    app.run(debug=True)