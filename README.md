# 🦴 BoneFractureAI - End-to-End Bone Fracture Detection System

A **full-stack AI-powered medical imaging system** that detects bone fractures from X-ray images using **Deep Learning (ResNet50 + PyTorch)** and provides **visual explanations via Grad-CAM**.

---

# 🚀 1. Project Overview

## 🎯 Objective

Build a **scalable AI system** that:

* Detects fractures from X-ray images
* Provides **confidence score**
* Highlights **important regions (Grad-CAM heatmap)**
* Stores prediction history
* Allows **model training + retraining**

---

## 🧠 What the Model Does

| Feature                 | Description                                       |
| ----------------------- | ------------------------------------------------- |
| **Classification**      | Predicts whether X-ray is `Normal` or `Fractured` |
| **Confidence Score**    | Probability of prediction                         |
| **Explainability**      | Uses Grad-CAM to show *why* prediction was made   |
| **Training Capability** | Train model on custom dataset                     |
| **Real-time Inference** | Instant prediction via API                        |

---

## 🔬 Model Pipeline (High-Level Flow)

```
X-ray Image → Preprocessing → ResNet50 → Prediction → Grad-CAM → Response
```

---

# 🏗️ 2. Tech Stack

## Backend

* **FastAPI** → API framework
* **PyTorch** → Deep learning
* **ResNet50** → Transfer learning model
* **Grad-CAM** → Model explainability
* **MongoDB** → Stores predictions
* **OpenCV + Pillow** → Image processing

## Frontend

* **React 19**
* **Tailwind CSS**
* **Shadcn UI**
* **Framer Motion**
* **React Dropzone**

---

# 📂 3. Project Structure

```
bone-fracture-ai/
├── backend/
│   ├── server.py              # API entry point
│   ├── model_manager.py       # Training + inference logic
│   ├── grad_cam.py            # Heatmap generation
│   ├── prepare_dataset.py     # Synthetic dataset generator
│   ├── requirements.txt
│   ├── .env
│   ├── data/
│   └── models/
│       └── best_model.pth
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── App.js
│   ├── package.json
│   └── .env
│
└── README.md
```

---

# ⚙️ 4. How the Model Works (Deep Explanation)

## Step 1: Preprocessing

* Resize image → `224x224`
* Normalize using ImageNet stats
* Convert to tensor

## Step 2: Transfer Learning

* Load **ResNet50 pretrained on ImageNet**
* Replace final layer:

```python
model.fc = nn.Linear(in_features, 2)
```

## Step 3: Training

* Loss: **CrossEntropyLoss**
* Optimizer: **Adam**
* Metrics: Accuracy, Loss

## Step 4: Prediction

* Softmax → Probability
* Class Output:

  * `0 → Normal`
  * `1 → Fractured`

## Step 5: Grad-CAM

* Extract last convolutional layer
* Compute gradients
* Generate heatmap overlay

---

# 🧪 5. Dataset

## Structure

```
data/
├── train/
│   ├── normal/
│   └── fractured/
└── val/
    ├── normal/
    └── fractured/
```

## Generate Sample Dataset

```bash
cd backend
python prepare_dataset.py
```

---

# 🏃 6. Local Setup (Complete Guide)

## 🔹 Prerequisites

* Python 3.11+
* Node.js 18+
* Yarn
* MongoDB
* 8GB RAM

---

## 🔹 Backend Setup

```bash
cd backend

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install fastapi uvicorn python-multipart pillow numpy scikit-learn opencv-python matplotlib seaborn motor python-dotenv pydantic
```

---

## 🔹 Backend Environment

```
MONGO_URL=mongodb://localhost:27017
DB_NAME=bone_fracture_db
CORS_ORIGINS=http://localhost:3000
```

---

## 🔹 Run Backend

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

---

## 🔹 Frontend Setup

```bash
cd frontend
yarn install
```

### Frontend `.env`

```
REACT_APP_BACKEND_URL=http://localhost:8001
```

---

## 🔹 Run Frontend

```bash
yarn start
```

---

# 🔄 7. Running the Application

## Option 1: Two Terminals

Terminal 1:

```bash
cd backend
source venv/bin/activate
uvicorn server:app --reload
```

Terminal 2:

```bash
cd frontend
yarn start
```

---

## Option 2: PM2 (Recommended)

```bash
npm install -g pm2
pm2 start ecosystem.config.js
pm2 logs
```

---

# 🧠 8. Training the Model

## Local Script Execution (Recommended for Custom Datasets)

You can directly run the training script from the backend directory to monitor logs in the terminal:

```bash
cd backend
source venv/bin/activate
python run_training.py
```

## API Call

```bash
curl -X POST http://localhost:8001/api/train \
-H "Content-Type: application/json" \
-d '{"epochs": 5, "batch_size": 16, "learning_rate": 0.0001}'
```

## UI Method

* Go to **Train Page**
* Set parameters
* Click **Start Training**

---

# 🔍 9. Prediction Flow

1. Upload image
2. Backend processes image
3. Model predicts class
4. Grad-CAM heatmap generated
5. Response sent to frontend

---

# 🔗 10. API Endpoints

| Endpoint               | Method | Description       |
| ---------------------- | ------ | ----------------- |
| `/api/`                | GET    | Health check      |
| `/api/predict`         | POST   | Predict fracture  |
| `/api/train`           | POST   | Train model       |
| `/api/training-status` | GET    | Training progress |
| `/api/model-metrics`   | GET    | Accuracy, loss    |
| `/api/dataset-info`    | GET    | Dataset stats     |

---

# 🐳 11. Docker Deployment

```bash
docker-compose up
```

Services:

* MongoDB
* Backend
* Frontend

---

# 🌐 12. Production Deployment

## Backend

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:8001
```

## Frontend

```bash
yarn build
```

## Best Practices

* Use **Nginx**
* Use **SSL**
* Use **MongoDB Atlas**
* Use **Docker / PM2**

---

# ⚠️ 13. Troubleshooting

## PyTorch Issue

```bash
pip install torch torchvision torchaudio
```

## MongoDB Issue

```bash
systemctl start mongodb
```

## Port Issue

```bash
lsof -ti:8001 | xargs kill -9
```

---

# 📊 14. Future Improvements

* Use real medical datasets
* Add multi-class fracture detection
* Deploy on cloud (AWS/GCP)
* Add user authentication
* Integrate AI reports

---

# ⚠️ 15. Disclaimer

This project is:

* ❌ NOT for real medical diagnosis
* ✅ For educational & research use only

---


---

# ⭐ Conclusion

BoneFractureAI is a **complete AI + Fullstack system** that demonstrates:

* Deep Learning (PyTorch)
* Explainable AI (Grad-CAM)
* Fullstack Integration (React + FastAPI)
* Scalable Architecture


---

🚀 *Ready to run locally and deploy in production!*
