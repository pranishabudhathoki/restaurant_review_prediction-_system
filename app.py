
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request
import pickle

# Here we are loading the Multinomial Naive Bayes model and CountVectorizer object from disk

filename = 'restaurant-sentiment-mnb-model6.pkl'

classifier = pickle.load(open(filename, 'rb'))


cv = pickle.load(open('cv-transform6.pkl', 'rb'))

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/')
def hello_world():
    return render_template("login.html")


database = {'aashika': '123', 'bipana': '1234', 'prani': '12345'}


@app.route('/form_login', methods=['POST', 'GET'])
def login():
    name1 = request.form['username']
    pwd = request.form['password']
    if name1 not in database:
        return render_template('login.html', info='Invalid User')
    else:
        if database[name1] != pwd:
            return render_template('login.html', info='Invalid Password')
        else:
            return render_template('index3.html', name=name1)
# def home():
#     return render_template('index3.html')


# @app.route('/contact')
# def contact():
#     return render_template('contact.html')


# @app.route('/about')
# def about():
#     return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
