from flask import Flask, render_template, request
import pickle
import os

# Load the saved model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        # Basic preprocessing: lowercase (add more if needed)
        text = news_text.lower()
        vect = vectorizer.transform([text])
        prediction = model.predict(vect)[0]
        label = 'True News' if prediction == 1 else 'Fake News'
        return render_template('index.html', prediction=label, news=news_text)

if __name__ == '__main__':
    print("Starting Flask app...")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug = True)
