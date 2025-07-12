from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import skin_cancer_detection as SCD
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model_skin_cancer.h5')

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def runhome():
    return render_template("home.html")

@app.route("/showresult", methods=["GET", "POST"])
def show():
    # Load and process image
    pic = request.files["pic"]
    inputimg = Image.open(pic).resize((28, 28))
    img = np.array(inputimg).reshape(-1, 28, 28, 3)

    # Make prediction
    prediction = model.predict(img)
    prediction = prediction.tolist()[0]

    # Get predicted class index
    class_ind = int(np.argmax(prediction))  # more stable than list.index()

    # Get class name
    result = SCD.classes.get(class_ind, "Unknown class")

    # Get class description
    descriptions = {
        0: "Actinic keratosis: A pre-malignant lesion or in situ squamous cell carcinoma.",
        1: "Basal cell carcinoma: A form of skin cancer most common on sun-exposed skin areas.",
        2: "Benign lichenoid keratosis: A type of benign skin lesion.",
        3: "Dermatofibroma: Noncancerous skin growths, usually firm and brownish.",
        4: "Melanocytic nevus: Commonly known as a mole, often benign.",
        5: "Pyogenic granulomas: Small, round, reddish growths, often bleed.",
        6: "Melanoma: A serious form of skin cancer, known to spread rapidly."
    }

    info = descriptions.get(class_ind, "Kindly visit the hospital, this type of class is found rarely.")

    return render_template("results.html", result=result, info=info)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
