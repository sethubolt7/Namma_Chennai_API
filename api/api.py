from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

url = input()

app = Flask(__name__)

@app.route('/run_script', methods=['POST'])
def run_script():
    # Get image data from request
    image_data = request.files['image']

    # saves the image data to a file
    image_path = 'uploaded_image.jpeg'
    image_data.save(image_path)

    # loading the image with OpenCV
    img = cv2.imread(image_path)

    # resize the image for predicting
    resize = tf.image.resize(img, (256, 256))

    # Load the models
    filter_1 = load_model("models\\filter_1_verify.h5")
    filter_2 = load_model("models\\filter_01_verify.h5")
    # decision_model = load_model("models\\decision_model.h5")

    # this makes the predictions
    pred_1 = filter_1.predict(np.expand_dims(resize/255, 0))
    pred_2 = filter_2.predict(np.expand_dims(resize/255, 0))
    # pred_3 = decision_model.predict(np.expand_dims(resize/255, 0))

    # class names
    # class_names1 = ["Class 0 unwanted", "Class 1 disaster"]
    class_names1 = ["notverified", "verified"] #1
    class_names2 = ["notverified", "verified"] #2
    # class_names3 = ["Class 0 flood issues", "Class 1 polluted", "Class 2 other water issues"]

    # this get's predicted classes
    predicted_class1 = class_names1[np.argmax(pred_1, axis=1)[0]]
    predicted_class2 = class_names2[np.argmax(pred_2, axis=1)[0]]
    # predicted_class3 = class_names3[np.argmax(pred_3, axis=1)[0]]

    #  this processes the flow
    result_text = ''
    if predicted_class1 == "verified":
        
        # result_text += f"{predicted_class1}"
        if predicted_class2 == "notverified":
            result_text += f"{predicted_class2}\n"
        #     result_text += f"decision_model says: {predicted_class3}\n\n"
        #     result_text += f"so it is :{predicted_class3}"
        else:
            result_text += f"{predicted_class2}\n"
        #     result_text += "so there is no water related issue found in the image"
    else:
        result_text += f"{predicted_class1}"
    print(pred_1)
    # Return the results as JSON to water_pred
    return jsonify({'result': result_text})

if __name__ == "__main__":
    app.run(debug=True, port=80)
