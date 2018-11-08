from flask import Flask, render_template, request

import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load('../data/bestModel.joblib')

@app.route('/')
def index():
    return render_template('category_form.html', codes=model.feature_names)

@app.route('/result', methods=['POST', ])
def result():
    # values = {}
    # for name in model.feature_names:
    #     values[name] = request.form.get(name)
    values = {name: request.form.get(name) for name in model.feature_names}    
    df = pd.DataFrame(values, index=[0, ])
    df = df.replace('', 0).astype(float)
    pct = df.divide(df.sum(axis=1), axis='rows').fillna(0)

    pred = model.predict(pct)

    html_args = dict(classes = ['table', 'thead-light'])

    return render_template('results.html', input=pct.T.to_html(**html_args), pred=pred)