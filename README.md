# 🌿 AgriVision — Crop Disease Detection & Advisory System

End-to-end AI system that detects crop diseases from leaf images and provides actionable recommendations.

---

## 🚀 Features

* **Multi-crop classification**: Tomato, Potato, Corn (9 classes)
* **Top-3 predictions** with confidence scores
* **User-friendly UI** (Streamlit) with image preview and progress bars
* **Actionable advice** based on predicted disease
* **FastAPI backend** for real-time inference
* **Robust evaluation** (confusion matrix, per-class metrics)

---

## 🧠 Model

* **Architecture**: Custom CNN (5 conv blocks)
* **Input**: 3×224×224 RGB
* **Dataset**: PlantVillage (curated subset)
* **Accuracy**: ~98% (validation)

---

## 🧩 Tech Stack

* **ML**: PyTorch
* **Backend**: FastAPI, Uvicorn
* **Frontend**: Streamlit
* **Utilities**: NumPy, scikit-learn, PIL

---

## 🏗️ Architecture

```
Streamlit UI  →  FastAPI (/predict)  →  Predictor (PyTorch)
                                       ├─ Softmax
                                       └─ Top-3 outputs
```

---

## 📸 Demo

> Add screenshots here (UI + Swagger)

* Upload image → get predictions + confidence + recommendation

---

## ▶️ How to Run

### 1) Clone & setup

```bash
git clone <your-repo-url>
cd agrivision
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Start backend

```bash
uvicorn backend.main:app --reload
```

### 3) Start UI

```bash
streamlit run frontend/app.py
```

Open:

* UI: http://localhost:8501
* API docs: http://127.0.0.1:8000/docs

---

## 📊 Example Output

```json
{
  "predictions": [
    {"label": "Tomato___Early_blight", "confidence": 0.9989},
    {"label": "Tomato___Late_blight", "confidence": 0.001},
    {"label": "Tomato___healthy", "confidence": 0.0001}
  ]
}
```

---

## 🧪 Evaluation

* Confusion matrix + classification report
* Strong performance across all classes
* Minor confusion between visually similar diseases

---

## 💡 Future Improvements

* EfficientNet fine-tuning
* Grad-CAM visual explanations
* Multi-language UI
* Batch inference

---

## 👤 Author

Sneha (AIML Engineer)

---

## ⭐ If you like this project

Give it a star and share feedback!
