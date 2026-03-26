# 🛒 AI-Based E-Commerce Recommendation System

![React](https://img.shields.io/badge/Frontend-React-blue?logo=react)
![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask)
![Python](https://img.shields.io/badge/Language-Python-yellow?logo=python)
![Machine Learning](https://img.shields.io/badge/ML-TF--IDF%20%7C%20Cosine%20Similarity-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Overview

An intelligent **AI-powered recommendation system** that suggests relevant products based on user input using **Machine Learning techniques**.

This project demonstrates real-world implementation of:

* Content-based filtering
* Natural Language Processing (TF-IDF)
* Full-stack integration (React + Flask)

---

## 🌐 Live Demo

🔗 **Frontend:** *Coming Soon*
🔗 **Backend API:** `http://127.0.0.1:5000`

> 💡 You can deploy using **Vercel (Frontend)** + **Render (Backend)** for a live portfolio project.

---

## 📸 Screenshots  

### 🔍 Search Interface  
![Search UI](screenshots/search.png)

### 🛍️ Recommendations Output  
![Results UI](screenshots/results.png)
## ✨ Features

* 🔍 Product search with real-time results
* 🤖 AI-based recommendation engine
* ⚡ TF-IDF powered similarity matching
* 🎯 Content-based filtering
* 🌐 Full-stack architecture
* 🖼️ Dynamic UI with product cards
* 🔄 API integration using Axios
* ⚠️ Error handling ("No product found")

---

## 🧠 Tech Stack

### Frontend

* React.js
* Axios
* CSS

### Backend

* Python
* Flask
* Pandas
* Scikit-learn

### Machine Learning

* TF-IDF Vectorization
* Cosine Similarity

---

## 📂 Project Structure

```bash id="struct2"
AI-Based-E-Commerce-Recommendation-System/
│
├── backend/
│   ├── app.py
│   ├── recommend.py
│   └── data/products.csv
│
├── frontend/
│   ├── src/
│   └── package.json
│
└── screenshots/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash id="clone2"
git clone https://github.com/your-username/AI-Based-E-Commerce-Recommendation-System.git
cd AI-Based-E-Commerce-Recommendation-System
```

---

### 2️⃣ Backend Setup

```bash id="backend2"
cd backend
pip install flask pandas scikit-learn flask-cors
python app.py
```

---

### 3️⃣ Frontend Setup

```bash id="frontend2"
cd frontend
npm install
npm start
```

---

## 🔗 API Endpoint

```http id="api2"
GET /recommend?product=<product_name>
```

### Example

```bash id="ex2"
http://127.0.0.1:5000/recommend?product=shoes
```

---

## ⚙️ Working Principle

* Combines product metadata (name + category + description)
* Converts text into vectors using TF-IDF
* Computes similarity using cosine similarity
* Returns top matching recommendations

---

## 📊 Resume Impact (ATS Keywords)

* Machine Learning
* Recommendation System
* Natural Language Processing
* Full Stack Development
* React.js
* Flask API
* Data Processing (Pandas)
* Scikit-learn

---

## 🚧 Future Enhancements

* 🔐 Authentication system
* 🛒 Cart & checkout
* 📊 Collaborative filtering
* ☁️ Cloud deployment
* 📱 Mobile responsive UI
* 🧠 Deep learning models

---

## 👩‍💻 Author

**Mamatha Takur**

* GitHub: https:https://github.com/TakurMamatha 
* LinkedIn: http://www.linkedin.com/in/takur 

---

## ⭐ Support

If you like this project:
⭐ Star this repository
🍴 Fork it
📢 Share it

---

## 📌 Conclusion

This project showcases the integration of **Machine Learning + Full Stack Development** to build intelligent applications. It is a strong portfolio project demonstrating real-world problem solving and system design.

---
