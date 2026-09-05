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

        # Input order MUST be the same as model training
        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Make prediction
        prediction = model.predict(input_data)

        # Get predicted species
        result = str(prediction[0]).strip()

        # Select the correct flower image
        image_files = {
            "iris-setosa": "setosa.jpg",
            "iris-versicolor": "versicolor.jpg",
            "iris-virginica": "virginica.jpg",
            "setosa": "setosa.jpg",
            "versicolor": "versicolor.jpg",
            "virginica": "virginica.jpg"
        }

        image_filename = image_files.get(
            result.lower(),
            "setosa.jpg"
        )

        # Send prediction, image name and measurements to HTML
        return render_template(
            "index.html",
            prediction=result,
            image_filename=image_filename,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid values in all fields."
        )


if __name__ == "__main__":
    app.run(debug=True)