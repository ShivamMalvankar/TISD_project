# 🌐 EchoChamberX: Bias & Polarization Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![NLP](https://img.shields.io/badge/NLP-NLTK%20VADER-orange)
![Frontend](https://img.shields.io/badge/Frontend-HTML%2FJS%2FChart.js-yellow)

**EchoChamberX** is a full-stack, real-time Natural Language Processing (NLP) dashboard designed to detect digital echo chambers, analyze media bias, and calculate public polarization on any given topic.

---

## 💡 The Problem

In today's digital landscape, algorithmic feeds often trap users in **echo chambers**, exposing them only to information that reinforces their existing beliefs.

This leads to:
- Increased polarization  
- Skewed public sentiment  
- Rapid spread of biased narratives  

**EchoChamberX** addresses this by providing a transparent, data-driven view of how topics are discussed across different platforms.

---

## ✨ Key Features

- 🔍 **Real-Time Data Aggregation**  
  Scrapes data from Google News, Bing (fallback), and Reddit.

- 🧠 **Sentiment Analysis (VADER)**  
  Classifies content into positive, negative, or neutral sentiment.

- ⚖️ **Bias Detection System**  
  Identifies whether content is **Pro**, **Anti**, or **Neutral**.

- 📊 **Polarization Score (0–100)**  
  Measures how divided opinions are on a topic.

- 🌐 **Interactive Dashboard UI**  
  Clean dark-mode interface with charts and insights.

---

## 🛠️ Tech Stack

### Backend
- Python  
- Flask  
- BeautifulSoup4  
- Requests  

### NLP & Data
- NLTK (VADER Sentiment Analyzer)  
- Pandas  

### Frontend
- HTML5 / CSS3  
- JavaScript (Vanilla)  
- Chart.js  

---

## ⚙️ How It Works

1. **User Input**  
   User enters a topic (e.g., *“Elections”*)

2. **Web Scraping (`webscarping.py`)**  
   - Collects articles and posts  
   - Cleans and extracts text  

3. **NLP Pipeline (`echochamberx.py`)**
   - Text preprocessing  
   - Sentiment analysis  
   - Bias classification  
   - Polarization score calculation  

4. **Visualization (`app.py + frontend`)**  
   - Data is sent as JSON  
   - Dashboard renders charts and insights  

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/EchoChamberX.git
cd EchoChamberX
