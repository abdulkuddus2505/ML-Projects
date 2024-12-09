from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the pre-trained model
with open('boston_model.pkl', 'rb') as file:
    model = pickle.load(file)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories='auto') 
original_data = pd.read_csv('boston.csv')
categorical_features = ['CHAS', 'RAD', 'TAX'] 
encoder.fit(original_data[categorical_features]) 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        CRIM = float(request.form['CRIM'])
        ZN = float(request.form['ZN'])
        INDUS = float(request.form['INDUS'])
        CHAS = float(request.form['CHAS'])
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        AGE = float(request.form['AGE'])
        DIS = float(request.form['DIS'])
        RAD = float(request.form['RAD'])
        TAX = float(request.form['TAX'])
        PT = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])


        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PT, B, LSTAT]],
                          columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PT', 'B', 'LSTAT'])
        
        # One-hot encode categorical features
        categorical_features = ['CHAS', 'RAD', 'TAX']
        encoded_features = encoder.transform(input_data[categorical_features])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        
        # Drop original categorical columns and add encoded columns
        input_data = input_data.drop(columns=categorical_features)
        input_data = pd.concat([input_data, encoded_df], axis=1)

        # Make prediction (input_data now has encoded features)
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)