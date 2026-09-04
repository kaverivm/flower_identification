
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from the HTML form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Create input data in the same order used during model training
        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Make prediction
        prediction = model.predict(input_data)

        # Get predicted species
        result = prediction[0]

        return render_template(
            "index.html",
            prediction=result
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid values in all fields."
        )


if __name__ == "__main__":
    app.run(debug=True)

