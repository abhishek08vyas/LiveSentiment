import os
import tweepy
from flask import Flask, request, render_template, jsonify
from sklearn.externals import joblib
from twitter import TwitterClient
import numpy as np
import nltk
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)
# Setup the client <query string, retweets_only bool, with_sentiment bool>    
api = TwitterClient('@Sirajology')
	
def strtobool(v):
    return v.lower() in ["yes", "true", "t", "1"]

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def results():
	if request.method == 'POST':
		review = request.form['review']
		review1 = review	
		corpus = []
		review = re.sub('[^a-zA-Z]', ' ', review)
		review = review.lower()
		review = review.split()
		lemmatizer = WordNetLemmatizer()
		review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
		review = ' '.join(review)
		corpus.append(review)
		classifier = joblib.load('classifier.pkl')
		tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')
		x_tfid = tfidfVectorizer.transform(corpus).toarray()
		answer = classifier.predict(x_tfid)
		answer = str(answer[0])
		pos = "That looks like a positive review."
		neg = "You dont seem to have liked that movie."
		if answer == '1':
			return render_template('algo.html', content=review1, prediction=pos)
		else:
			return render_template('algo.html', content=review1, prediction=neg)
	return render_template('algo.html', form=form)
	
@app.route('/about')
def team():
    return render_template('team.html')

@app.route('/algo')
def algo():
    return render_template('algo.html')

@app.route('/tweets')
def tweets():
        retweets_only = request.args.get('retweets_only')
        api.set_retweet_checking(strtobool(retweets_only.lower()))
        with_sentiment = request.args.get('with_sentiment')
        api.set_with_sentiment(strtobool(with_sentiment.lower()))
        query = request.args.get('query')
        api.set_query(query)

        tweets = api.get_tweets()
        return jsonify({'data': tweets, 'count': len(tweets)})


if __name__ == '__main__':
    #model = joblib.load('logisticreg.pkl')
    app.run(host='127.0.0.1', port=8050, debug=True)
