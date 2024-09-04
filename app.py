from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import string

app = Flask(__name__)

@app.route('/', methods=['GET'])

def index():
    return render_template('index.html')

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

def get_sentence_vector(sentence, model):
    words = preprocess_text(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
@app.route('/', methods=['POST'])
def predict():
    review_text = request.form.get('review_text')
    pickle_w2v = open('w2v.sav', 'rb')
    tokens = preprocess_text(review_text)
    vectors = [get_sentence_vector(tokens, pickle.load(pickle_w2v))]
    pickle_model = open('lrmodel.sav', 'rb')
    model = pickle.load(pickle_model)
    prediction = model.predict(vectors)
    if prediction[0] == -1:
        return render_template('negative.html', prediction='Negative')
    else:
        return render_template('positive.html', prediction='Positive')
    #return render_template('index.html')

if __name__ == "__main__":
    app.run(port = 3000, debug=True)