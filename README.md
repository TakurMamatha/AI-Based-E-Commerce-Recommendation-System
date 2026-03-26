# 🛒 AI-Based E-Commerce Recommendation System

An intelligent product recommendation system that suggests similar products based on user input using **Machine Learning (TF-IDF + Cosine Similarity)**. This project demonstrates how AI can enhance user experience in e-commerce platforms by delivering personalized suggestions.

---

## 🚀 Features

* 🔍 Search products dynamically
* 🤖 AI-based recommendation engine
* ⚡ Fast similarity matching using TF-IDF
* 🎯 Content-based filtering (name + category + description)
* 🌐 Full-stack integration (React + Flask)
* 🖼️ Dynamic product display with images
* 🔄 Real-time API communication

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

```
AI-Based-E-Commerce-Recommendation-System/
│
├── backend/
│   ├── app.py
│   ├── recommend.py
│   └── data/
│       └── products.csv
│
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── index.js
│   │   └── App.css
│   └── package.json
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Based-E-Commerce-Recommendation-System.git
cd AI-Based-E-Commerce-Recommendation-System
```

---

### 🔹 2. Backend Setup (Flask)

```bash
cd backend
pip install flask pandas scikit-learn flask-cors
python app.py
```

Server will run at:

```
http://127.0.0.1:5000
```

---

### 🔹 3. Frontend Setup (React)

```bash
cd frontend
npm install
npm start
```

Frontend runs at:

```
http://localhost:3000
```

---

## 🔗 API Endpoint

### Get Recommendations

```
GET /recommend?product=<product_name>
```

### Example:

```
http://127.0.0.1:5000/recommend?product=shoes
```

### Response:

```json
[
  "running shoes",
  "sneakers",
  "t-shirt"
]
```

---

## 🧩 How It Works

1. Combines product **name, category, and description**
2. Converts text data into numerical vectors using **TF-IDF**
3. Computes similarity using **Cosine Similarity**
4. Returns top matching products based on input

---

## 📸 UI Preview

* Clean and responsive interface
* Product cards with images and pricing
* Loading state handling
* Error handling (No product found)

---

## 🛠️ Improvements & Future Scope

* 🔐 User authentication (Login/Signup)
* 🛒 Add to Cart & Checkout functionality
* 📊 Advanced recommendation (collaborative filtering)
* ☁️ Deployment (AWS / Vercel / Render)
* 🧠 Deep Learning-based recommendations
* 📱 Mobile responsive design

---

## 🧑‍💻 Author

**Mamatha Takur**

* GitHub: https://github.com/your-username
* LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ Acknowledgements

* Scikit-learn documentation
* React official docs
* Flask framework

---

## 📌 Conclusion

This project showcases how **AI and Web Development** can be combined to build smart, user-centric applications. It is a strong demonstration of full-stack development with machine learning integration.

---

⭐ If you like this project, consider giving it a star!
