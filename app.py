from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
import os, json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from deep_translator import GoogleTranslator
import requests

app = Flask(__name__)
# ---------------------------------
# Step 1: Auto-download model if missing
# ---------------------------------
MODEL_PATH = "leaf_disease_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1mak1fsuHTJ7yNJW5i3Qonz-HLKwWb75x"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file from Google Drive...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists — skipping download.")

# Call the function before loading the model
download_model()


# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = "leaf_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open("class_indices.json", "r") as f:
    classes = list(json.load(f).keys())

# Load remedies dataset
df = pd.read_csv("complete_disease_remedies.csv")

# -------------------------
# Disease name translations (English → Marathi)
# -------------------------
disease_translations = {
    "Diseased Brinjal Leaf": "रोगट वांगी पान",
    "Early_blight_leaves": "लवकर करपा पाने",
    "Fresh Brinjal Leaf": "ताजी वांगी पान",
    "Healthy_leaves": "निरोगी पाने",
    "Late_blight_leaves": "उशीरा करपा पाने",
    "Pepper__bell___Bacterial_spot": "मिरची जिवाणू डाग",
    "Pepper__bell___healthy": "निरोगी मिरची",
    "Tomato_Bacterial_spot": "टोमॅटो जिवाणू डाग",
    "Tomato_Early_blight": "टोमॅटो लवकर करपा",
    "Tomato_Late_blight": "टोमॅटो उशीरा करपा",
    "Tomato_Leaf_Mold": "टोमॅटो पान बुरशी",
    "Tomato_Septoria_leaf_spot": "टोमॅटो सेप्टोरिया पान डाग",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "टोमॅटो लाल कोळी",
    "Tomato__Target_Spot": "टोमॅटो टार्गेट डाग",
    "Tomato_YellowLeaf__Curl_Virus": "टोमॅटो पिवळे पान वळण विषाणू",
    "Tomato_mosaic_virus": "टोमॅटो मोझॅक विषाणू",
    "Tomato_healthy": "निरोगी टोमॅटो",
    "diseased okra leaf": "रोगट भेंडी पान",
    "fresh okra leaf": "ताजी भेंडी पान",
    "healthy": "निरोगी",
    "leaf curl": "पाने वाकणे",
    "leaf spot": "पानांवर डाग",
    "whitefly": "पांढरी माशी",
    "yellowish": "पिवळसर"
}

# -------------------------
# Remedy table headers
# -------------------------
table_headers = {
    "en": ["Type", "Pesticide", "Dosage", "Solution", "Amount", "Frequency", "Notes", "Safety Notes"],
    "hi": ["प्रकार", "कीटनाशक", "खुराक", "उपाय", "मात्रा", "आवृत्ति", "टिप्पणियाँ", "सुरक्षा नोट्स"],
    "mr": ["प्रकार", "कीटकनाशक", "डोस", "उपाय", "प्रमाण", "वारंवारता", "नोंदी", "सुरक्षा नोंदी"]
}

# -------------------------
# Helper functions
# -------------------------
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def normalize(text):
    return str(text).strip().lower().replace(" ", "_")

def translate_text(text, target_lang):
    if target_lang == "en" or text == "N/A":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(str(text))
    except Exception:
        return text

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    lang = request.form.get("lang", "en")

    # Save uploaded file
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # Predict disease
    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))
    predicted_class = classes[class_idx]

    # Translate label
    if lang == "mr":
        translated_label = disease_translations.get(predicted_class, predicted_class)
    else:
        translated_label = translate_text(predicted_class.replace("_", " "), lang)

    remedies = []
    headers = table_headers.get(lang, table_headers["en"])

    # -------------------------------
    # Confidence-based logic
    # -------------------------------
    if confidence < 0.5:
        # 🟡 Low confidence → General preventive
        remedies = [{
            "Type": translate_text("General Preventive", lang),
            "Pesticide": translate_text("Neem oil", lang),
            "Dosage": "5 ml/litre",
            "Solution": translate_text("Spray weekly to prevent fungal or viral infection", lang),
            "Amount": "300 litres/acre",
            "Frequency": "Weekly",
            "Notes": translate_text("Image unclear or unrelated — applied general prevention.", lang),
            "Safety Notes": translate_text("Safe, eco-friendly.", lang)
        }]
        predicted_label = translate_text("Low confidence — not a clear leaf", lang)

    elif confidence < 0.8:
        # 🟠 Medium confidence → Only organic/traditional remedies
        rows = df[df["disease"].apply(normalize) == normalize(predicted_class)]

        if not rows.empty:
            organic_traditional = rows[
                rows["remedy_type"].str.lower().isin(["organic", "traditional"])
            ]

            if not organic_traditional.empty:
                for _, r in organic_traditional.iterrows():
                    remedy = {
                        "Type": translate_text(r.get("remedy_type", "N/A"), lang),
                        "Pesticide": translate_text(r.get("pesticide", "N/A"), lang),
                        "Dosage": f"{r.get('dosage_value', 'N/A')} {r.get('dosage_unit', '')}".strip(),
                        "Solution": translate_text(r.get("solution", "N/A"), lang),
                        "Amount": translate_text(r.get("amount", "N/A"), lang),
                        "Frequency": translate_text(r.get("frequency", "N/A"), lang),
                        "Notes": translate_text(r.get("notes", "N/A"), lang),
                        "Safety Notes": translate_text(r.get("safety_notes", "N/A"), lang),
                    }
                    remedies.append(remedy)
            else:
                remedies.append({
                    "Type": translate_text("Organic", lang),
                    "Pesticide": translate_text("Neem oil", lang),
                    "Dosage": "5 ml/litre",
                    "Solution": translate_text("General organic preventive spray", lang),
                    "Amount": "300 litres/acre",
                    "Frequency": "Weekly",
                    "Notes": translate_text("No organic or traditional remedies found for this disease.", lang),
                    "Safety Notes": translate_text("Safe, eco-friendly.", lang)
                })
        else:
            remedies.append({
                "Type": translate_text("Organic", lang),
                "Pesticide": translate_text("Neem oil", lang),
                "Dosage": "5 ml/litre",
                "Solution": translate_text("General preventive organic spray", lang),
                "Amount": "300 litres/acre",
                "Frequency": "Weekly",
                "Notes": translate_text("Disease not found in dataset.", lang),
                "Safety Notes": translate_text("Safe, eco-friendly.", lang)
            })

        predicted_label = f"{translated_label} ({translate_text('Medium confidence', lang)})"

    else:
        # 🟢 High confidence → Full remedy list
        rows = df[df["disease"].apply(normalize) == normalize(predicted_class)]
        if not rows.empty:
            for _, r in rows.iterrows():
                remedy = {
                    "Type": r.get("remedy_type", "N/A"),
                    "Pesticide": r.get("pesticide", "N/A"),
                    "Dosage": f"{r.get('dosage_value', 'N/A')} {r.get('dosage_unit', '')}".strip(),
                    "Solution": r.get("solution", "N/A"),
                    "Amount": r.get("amount", "N/A"),
                    "Frequency": r.get("frequency", "N/A"),
                    "Notes": r.get("notes", "N/A"),
                    "Safety Notes": r.get("safety_notes", "N/A"),
                }
                for key in remedy:
                    remedy[key] = translate_text(remedy[key], lang)
                remedies.append(remedy)
        predicted_label = f"{translated_label} ({translate_text('High confidence', lang)})"

    # -------------------------------
    # Render template
    # -------------------------------
    return render_template(
        "result.html",
        image_file=f"uploads/{file.filename}",
        predicted_label=predicted_label,
        confidence=round(confidence * 100, 2),
        remedies=remedies,
        headers=headers,
        selected_language=lang
    )


if __name__ == "__main__":
    app.run(debug=True)
