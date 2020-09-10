from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
import gensim
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.remove("not")
import random
from textblob import TextBlob
from textblob import Word


app = Flask(__name__)
MODEL_PATH1 = 'models/tfidf_model.pkl'
MODEL_PATH = 'models/lr_model.pkl'

@app.route('/')
def home():
 return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if(len(message)>2):
            text = message
            pre_processed_reviews = []
            data = gensim.utils.simple_preprocess(text, min_len=2)
            review = ' '.join(WordNetLemmatizer().lemmatize(word) for word in data if word not in stop_words)
            pre_processed_reviews.append(review.strip())
            tfidf_model = load_model(MODEL_PATH1)
            vect = tfidf_model.transform(pre_processed_reviews)
            lr_model = load_model(MODEL_PATH)
            my_prediction = lr_model.predict(vect)
        else:
            my_prediction=3
            return render_template('home.html',prediction = my_prediction)
            
        blob=TextBlob(text)
        nouns=list()
        for word,tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
        display=[]
        output=""
        for item in random.sample(nouns,len(nouns)):  
            word=Word(item)
            if word not in display:
                display.append(word.capitalize())
                
        for i in display:
            if len(i) > 2:
                output = output + " " + i
        
        return render_template('home.html',prediction = my_prediction, summary = output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
