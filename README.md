# 🧠 Mapping Customer Satisfaction Dimensions in the Essential Oils Market Using Explainable AI

This repository contains the full pipeline and documentation for the research project **"Mapping Essential Oil Customer Satisfaction Dimensions from Online Reviews Using Explainable AI and Kano Model"**, published in *Spektrum Industri Vol. 23 No. 1 (2025)*.

## 🧩 Project Summary

In this project, I developed an **end-to-end framework** to analyze Indonesian e-commerce reviews of essential oil products. The goal was to **identify key customer satisfaction dimensions (CSDs)** directly from customer-generated content using **advanced NLP and machine learning techniques**, enabling companies to improve their offerings without relying on traditional surveys.

---

## 📦 Key Components

### 📌 1. Data Collection
- Source: Tokopedia reviews
- Tool: Python (Selenium + BeautifulSoup)
- Output: 23,068 unique reviews cleaned and segmented into **low**, **mid**, and **high-price tiers**

### 🧠 2. Topic Modeling (BERTopic)
- Extracted semantic topics from unstructured reviews
- Manually mapped 11 key dimensions (e.g., Aroma, Price, Packaging, Delivery, Efficacy)

### 🔍 3. Sentiment Labeling (GPT-4o-mini)
- Each review was labeled with **dimension-level sentiment** using a prompt-based approach
- Output used for training machine learning models

### 📈 4. Predictive Modeling (XGBoost Regression)
- Features: one-hot encoded CSD-sentiment pairs
- Labels: review ratings
- Metrics: Achieved **MAE = 0.1251** on test set

### ⚙️ 5. Explainability (SHAP)
- SHAP values used to interpret model predictions
- Quantified positive and negative contributions of each dimension per segment

### 🧭 6. Kano Classification (Based on SHAP)
- Attributes categorized into:
  - **Performance**: Aroma, Quality, Packaging
  - **Must-be**: Price (mid-tier), Diffuser Support (premium-tier)
  - **Latent Excitement**: Delivery (low-tier)

---

## 🛠️ Tools Used
- `Python`
  - `Selenium`, `BeautifulSoup` for scraping
  - `BERTopic`, `scikit-learn`, `XGBoost`, `SHAP`, `pandas`
- `OpenAI API` (GPT-4o-mini)
- `Matplotlib / Seaborn` for visuals
---

## 🧪 Key Insights

- **Performance Attributes Dominate**: Across all segments, tangible product features like aroma and quality were strong drivers of satisfaction.
- **Price Sensitivity in Mid-tier**: Price was a must-be condition—its absence immediately led to dissatisfaction.
- **Excitement Opportunities**: In the low-price segment, improving delivery or efficacy created outsized gains in satisfaction, even without expectation.
- **Explainable AI** bridged the gap between ML predictions and business interpretability, enhancing the strategic value of review data.

