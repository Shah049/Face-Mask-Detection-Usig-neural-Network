from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = os.path.join("model", "spam_model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email_text"]
        prediction = predict_spam(email_text)
        return render_template("index.html", email_text=email_text, prediction=prediction)
    return render_template("index.html")

def predict_spam(email_text):
    # Transform the email text into numeric features using the vectorizer
    email_text_transformed = vectorizer.transform([email_text])  # Transform into numeric features

    # Predict using the model
    prediction = model.predict(email_text_transformed)[0]
    if prediction == 1:
        return "?? Spam Email"
    else:
        return "? Not Spam"

if __name__ == "__main__":
    app.run(debug=True)
