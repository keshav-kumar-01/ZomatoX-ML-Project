# 🍽️ ZomatoX: Machine Learning & Trend Analysis on Restaurant Data

ZomatoX is a full-stack data science project that uses enhanced restaurant data from Zomato to analyze trends, build ML models, and power a real-world restaurant recommendation engine.

## 📌 Dataset Info
- **Source**: Enhanced Zomato Dataset (Kaggle)
- **Rows**: 123,000+
- **Features**: Ratings, prices, cuisines, cities, bestseller flags, engineered metrics

## 🧠 Modules Built

### 1️⃣ Trend Analysis Dashboard
- Visualizes how cuisines perform across different Indian cities.
- Heatmaps for ratings, pricing, and popularity.

### 2️⃣ Menu Item Price Predictor
- ML model trained on 18+ features.
- Predicts INR price of menu items using Random Forest.
- Achieved **R² = 0.9999**.

### 3️⃣ Restaurant Success Classifier
- Flags successful restaurants using rating + popularity thresholds.
- Achieved **91% accuracy** using Random Forest Classifier.

### 4️⃣ Recommendation System
- Content-based system suggests restaurants based on city + cuisine.
- Sortable by rating, price, or value-for-money.

### 5️⃣ Deployment-Ready Exports
- Saved models (`.pkl`), restaurant profiles, and preprocessed data
- Optional Streamlit UI available in `/streamlit_app/`

## 🚀 Tech Stack
- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- Plotly (for interactive trend analysis)
- Jupyter Notebooks
- Streamlit (UI-ready, optional)

## 📁 Project Structure
│
├── notebooks/
├── models/
├── data/
├── visuals/
├── streamlit_app/
├── summary_report.md
└── README.md


## 📊 Sample Output

### 📌 Cuisine Trends
![Cuisine Trends](visuals/plot_rating_distribution.png)

### 📌 Feature Importance
![Feature Importance](visuals/plot_price_feature_importance.png)

---

## 🧑‍💻 Author

Made with ❤️ by [Your Name](https://www.linkedin.com/in/your-profile)

## 📜 License
CC0: Public Domain — feel free to use, fork, or build upon this project.
