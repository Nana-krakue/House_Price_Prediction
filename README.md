# 🏠 House Price Prediction API

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deploy-Render-green.svg)](https://render.com)

---

## 📌 Overview
House Price Prediction is a machine learning project that provides an API for predicting housing prices based on property features such as bedrooms, bathrooms, and square footage.  

The project was developed as part of the **ML Zoomcamp** course and is deployed on **Render** using Flask and Gunicorn inside a Docker container.

---

## ⚙️ Features
- RESTful API built with **Flask**  
- Model trained with **scikit-learn**  
- Predictions served via `/predict` endpoint  
- Dockerized for easy deployment  
- Hosted on **Render** with auto-deploy from GitHub  

---

## 📂 Project Structure
House_Price_Prediction/

│

├── app.py                   # Flask API

├── requirements.txt         # Python dependencies

├── Dockerfile              # Container setup

├── models/

│   └── house_price_model.pkl   # Trained ML model

└── README.md                # Project documentation


---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Nana-krakue/House_Price_Prediction.git
cd House_Price_Prediction
```
### 2. Install dependencies
```bash

pip install -r requirements.txt
```
### 3. Run locally
```bash
python app.py
```
The API will be available at http://127.0.0.1:5000.
### 🐳 Docker Setup
### Build the image
```bash
docker build -t house-price-api .
```
### Run the container
```bash
docker run -p 5000:5000 house-price-api
```
### 🚀 Deployment on Render
1. Push your repo to GitHub.
2. Create a Web Service on Render.
3. Render auto-detects your Dockerfile.
4. Ensure your Dockerfile has:

```dockerfile
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
```
5. Render builds and deploys automatically.
6. You’ll get a public URL like:

```Code
https://house-price-api.onrender.com
```
📡 API Usage
Endpoint: /predict
Method: POST  
Content-Type: application/json

Example Request
```bash
curl -X POST https://house-price-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"bedrooms":3,"bathrooms":2,"sqft_living":1500}'
```
Example Response
```json
{
  "predicted_price": 350000.0
  }
```
### 🔧 Troubleshooting
Error: '' is not a valid port number  
→ Ensure app.py uses:

```python
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
```
Error: Connection in use: ('0.0.0.0', 5000)  
→ Stop any process already using port 5000 or run Gunicorn on another port locally.

Error: COPY models/ models/ not found  
→ Make sure the models/ folder exists and contains your .pkl file.
### 📚 Tech Stack
Flask

Gunicorn

scikit-learn

pandas / numpy

Docker

Render

👩‍💻 Author
Developed by Nana Krakue as part of the ML Zoomcamp course.
