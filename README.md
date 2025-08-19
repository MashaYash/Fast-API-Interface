# 🏠 House Price Prediction API

A FastAPI-based web service that uses a trained machine learning model to predict house prices based on user input.

---

## 📌 Problem Description

The goal is to build a regression model that predicts the price of a house based on features like:

- Area (square footage)
- Number of bedrooms
- Number of bathrooms
- Number of stories

A secondary model also uses all available features from a housing dataset to compare performance.

---

## ⚙️ Model Choice Justification

**RandomForestRegressor** was selected because:

- It handles nonlinear relationships well.
- It's robust to outliers and overfitting (with tuning).
- Works effectively with both numerical and boolean/categorical features.
- Performs well on tabular datasets like housing.

Hyperparameter tuning with `RandomizedSearchCV` was used to improve performance.

---

## 🚀 How to Run the Application

### 🔧 1. Clone or download this repository

```bash
git clone https://your-repo-url
cd your-project-folder
