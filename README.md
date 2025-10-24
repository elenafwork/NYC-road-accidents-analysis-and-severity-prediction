# 🚦 NYC Road Accidents: Analysis and Severity Prediction

This project analyzes motor vehicle collision data from New York City to identify key factors contributing to crash severity and to build predictive models that classify accidents by their severity level (minor, injury, fatal).  
The goal is to support data-driven safety planning and resource allocation.

---

## 📊 Project Overview

New York City has one of the largest urban traffic networks in the world, leading to thousands of motor vehicle accidents annually.  
This project explores historical crash data to uncover:

- Spatial and temporal patterns of collisions  
- High-risk factors (weather, time of day, vehicle type, etc.)  
- The relationship between driver behavior and crash outcomes  
- Predictive modeling of **crash severity**

---

## 🧰 Tech Stack

**Languages:** Python (pandas, NumPy, scikit-learn, matplotlib, seaborn)  
**Modeling:** Logistic Regression, Random Forest, XGBoost, SMOTE (for class imbalance)  
**Visualization:** Seaborn, Plotly, Folium (for geospatial maps)  
**Data Source:** [NYC Open Data – Motor Vehicle Collisions](https://data.cityofnewyork.us/)

---

## 📂 Repository Structure

NYC-road-accidents-analysis-and-severity-prediction/
│
├── data/ # Raw and cleaned datasets
├── notebooks/
│ ├── 01_data_cleaning.ipynb # Handling missing values, duplicates, outliers
│ ├── 02_eda_visualization.ipynb # Exploratory analysis, trends, correlations
│ ├── 03_modeling.ipynb # Logistic Regression, Random Forest, XGBoost
│ ├── 04_model_evaluation.ipynb # Confusion matrices, ROC curves, metrics
│
├── outputs/
│ ├── figures/ # Plots, correlation heatmaps, ROC curves
│ └── models/ # Saved model files (.pkl)
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation (you are here)
└── LICENSE


---

## 🧹 Data Cleaning & Preprocessing

1. **Removed duplicates and irrelevant columns**  
2. **Handled missing coordinates** and filtered invalid entries  
3. **Encoded categorical features** (e.g., vehicle type, borough, weather)  
4. **Normalized/standardized numeric variables**  
5. **Balanced dataset** using SMOTE to handle minority classes  

---

## 📈 Exploratory Data Analysis (EDA)

- Accident counts by **borough**, **hour of day**, and **day of week**  
- Correlation between **light conditions**, **weather**, and **severity**  
- Heatmaps and scatterplots to visualize cluster patterns  
- Mapping high-density crash areas with **Folium**

---

## 🤖 Modeling & Evaluation

| Model | Accuracy | Precision | Recall | AUC |
|:------|:---------:|:----------:|:-------:|:----:|
| Logistic Regression | 0.73 | 0.70 | 0.68 | 0.75 |
| Random Forest | 0.78 | 0.77 | 0.75 | 0.81 |
| XGBoost | **0.81** | **0.80** | **0.78** | **0.84** |

**Key insights:**
- **Weather, time of day, and driver inattention** are major predictors.  
- **XGBoost** achieved the best performance and generalization.  
- SMOTE improved minority-class recall substantially.

---

## 🧭 Recommendations

- Prioritize **evening patrols** and **visibility measures** in boroughs with high night-time crash rates.  
- Automate **duplicate record detection** and enforce **standardized input formats** during case intake.  
- Integrate **geospatial dashboards** for real-time monitoring of severity hotspots.

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/elenafwork/NYC-road-accidents-analysis-and-severity-prediction.git
   cd NYC-road-accidents-analysis-and-severity-prediction




👩‍💻 Author

Elena Fadeeva
Master’s in Analytics | Northeastern University
📧 LinkedIn
 | GitHub

📜 License

This project is licensed under the MIT License
.

📚 References

NYC Open Data: Motor Vehicle Collisions

scikit-learn Documentation: https://scikit-learn.org/
