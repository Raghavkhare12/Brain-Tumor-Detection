from flask import Flask, request, render_template
import numpy as np
import os
import json
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

model = load_model("model/final_model.h5")

for layer in model.layers:
    print(layer.name)
    
with open("model/class_labels.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    idx = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    return class_names[idx], confidence, img_array

def interpret_result(label):
    if label == "notumor":
        return "No Tumor Detected"
    return f"Tumor Detected ({label})"

def doctor_advice(label, severity):

    if label == "notumor":
        return "No tumor detected. Routine monitoring is recommended."

    base_msg = {
        "glioma": "Glioma detected. Requires neurological evaluation.",
        "meningioma": "Meningioma detected. Usually benign but should be monitored.",
        "pituitary": "Pituitary tumor detected. May affect hormonal balance."
    }

    severity_msg = {
        "Low": "Model attention is low. This may indicate early-stage or small affected region.",
        "Moderate": "Moderate activation observed. Further diagnostic imaging is advised.",
        "High": "High activation detected. Immediate medical consultation is strongly recommended."
    }

    return f"{base_msg.get(label, '')} {severity_msg.get(severity, '')}"

def get_gradcam(img_array, last_conv_layer_name="conv5_block3_out"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def get_severity(heatmap):
    heatmap = heatmap / (heatmap.max() + 1e-8)
    score = np.mean(heatmap)

    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    else:
        return "High"

def save_gradcam(img_path, heatmap, filename):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join("static/heatmaps", filename)
    cv2.imwrite(save_path, superimposed_img)

    return save_path    

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files["file"]
        filename = file.filename

        upload_path = os.path.join("static/uploads", filename)
        file.save(upload_path)

        label, confidence, img_array = predict(upload_path)
        result = interpret_result(label)
        if label == "notumor":
            color = "lightgreen"
        else:
            color = "red"
        heatmap = get_gradcam(img_array)
        heatmap_path = save_gradcam(upload_path, heatmap, filename)

        severity = get_severity(heatmap)

        advice = doctor_advice(label, severity)

        confidence = round(confidence, 2)

        return render_template(
            "index.html",
            prediction=result,
            confidence=round(confidence, 3),
            image_path=upload_path,
            heatmap_path=heatmap_path,
            color=color,
            advice=advice,
            severity=severity
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))