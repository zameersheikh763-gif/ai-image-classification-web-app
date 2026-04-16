from flask import Flask, request, render_template_string
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)


app = Flask(__name__)

# Load pre-trained model
model = MobileNetV2(weights="imagenet")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Classifier</title>
    <style>
        body{
            font-family: Arial;
            text-align:center;
            margin-top:50px;
            background:#f4f4f4;
        }
        .box{
            width:400px;
            margin:auto;
            padding:30px;
            background:white;
            border-radius:10px;
            box-shadow:0 0 10px gray;
        }
        h1{
            color:#333;
        }
        button{
            padding:10px 20px;
            background:blue;
            color:white;
            border:none;
            border-radius:5px;
            cursor:pointer;
        }
        img{
            margin-top:20px;
            width:250px;
            border-radius:10px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1>AI Image Classifier</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <br><br>
            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <h2>Prediction: {{ prediction }}</h2>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]

        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        prediction = decode_predictions(preds, top=1)[0][0][1]

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)

    
