from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import gensim
import xgboost as xgb
import re

# Load Word2Vec model
word2vec_model = gensim.models.Word2Vec.load("word2vec.model")

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Preprocess tweet text
def preprocess(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z\s]", "", text)
    return text.lower().strip()

# Convert tweet to vector using word2vec
def document_vector(doc):
    words = [word for word in doc.split() if word in word2vec_model.wv.index_to_key]
    if words:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        tweet=request.form['tweet']
        cleaned = preprocess(tweet)
        vector = document_vector(cleaned)
        dmatrix = xgb.DMatrix([vector])
        pred = model.predict(dmatrix)
        final_pred = int(np.argmax(pred))

        sentiment = "Normal" if final_pred == 1 else "Depression Detected"
        return render_template('index.html', prediction_text='Prediction: {}'.format(sentiment),tweet_text=tweet)

    except Exception as e:
        return render_template('index.html', prediction_text='Error!! Enter tweet again.')

if __name__ == "__main__":
    app.run(debug=True)
