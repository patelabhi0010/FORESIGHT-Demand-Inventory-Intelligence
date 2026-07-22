# 🛍️ FORESIGHT – AI-Powered Retail Demand Forecasting & Inventory Intelligence Platform

FORESIGHT is an end-to-end **Retail Data Analytics** project that predicts future product demand, analyzes inventory risks, and provides interactive business insights using **Python, Machine Learning, and Power BI**.

The project demonstrates the complete data analytics workflow, including data generation, preprocessing, feature engineering, demand forecasting, inventory risk analysis, and dashboard visualization.

---

# 📌 Project Objectives

- Generate synthetic retail datasets.
- Build an automated ETL pipeline.
- Perform feature engineering for demand prediction.
- Forecast future product demand using Machine Learning.
- Identify stockout and overstock risks.
- Create an interactive Power BI dashboard.
- Support data-driven inventory decisions.

---

# 🚀 Project Workflow

```
Retail Data Generation
          │
          ▼
     ETL Pipeline
(Data Cleaning & Merging)
          │
          ▼
 Feature Engineering
          │
          ▼
 Machine Learning Model
(Random Forest Regressor)
          │
          ▼
 Demand Forecast
          │
          ▼
 Inventory Risk Analysis
          │
          ▼
 Power BI Dashboard
```

---

# 📂 Project Structure

```
FORESIGHT/
│
├── anaconda_projects/
│   └── db/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
│
├── models/
│
├── notebook/
│   ├── EDA.ipynb
│   └── Baseline_Model.ipynb
│
├── reports/
│
├── src/
│   ├── pipeline.py
│   ├── features.py
│   ├── forecast.py
│   ├── risk.py
│
├── dashboard.html
├── foresight dashboard.pbix
├── generate_data.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The project generates synthetic retail datasets that simulate real business operations.

Main datasets include:

- SKU Master
- Daily Sales
- Inventory Snapshots
- Calendar
- Final Dataset
- Feature Engineered Dataset
- Demand Forecast
- Inventory Risk Report

---

# ⚙️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Data Processing |
| Pandas | Data Cleaning |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| Jupyter Notebook | Data Analysis |
| Power BI | Dashboard Development |
| DAX | KPI Calculations |
| CSV | Data Storage |

---

# 🔄 ETL Pipeline

The ETL pipeline performs:

- Data Extraction
- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Date Formatting
- Dataset Merging
- Final Dataset Creation

---

# 📈 Feature Engineering

The project creates several important features including:

- Month
- Week
- Quarter
- Weekend Flag
- Lag Sales
- Rolling Mean
- Rolling Standard Deviation
- Promotion Flag
- Holiday Flag
- Inventory Coverage
- Price Features
- Discount Percentage

These features improve forecasting accuracy.

---

# 🤖 Machine Learning

### Algorithm Used

- Random Forest Regressor

### Model Evaluation

- MAE
- RMSE
- MAPE
- WAPE
- R² Score

The trained model predicts future demand for every product based on historical sales data.

---

# 📦 Inventory Risk Analysis

The project identifies:

- Stockout Risk
- Overstock Risk
- Inventory Health
- Reorder Recommendations
- Weeks of Inventory

These insights help retailers optimize inventory planning.

---

# 📊 Power BI Dashboard

The Power BI dashboard provides interactive reports including:

- Executive Dashboard
- Sales Analysis
- Demand Forecast
- Inventory Dashboard
- Risk Analysis
- Product Performance

Dashboard KPIs include:

- Total Sales
- Revenue
- Forecast Demand
- Inventory Status
- High Risk Products

---

# 📁 Output Files

The project generates the following outputs:

- Cleaned Dataset
- Feature Engineered Dataset
- Demand Forecast Results
- Inventory Risk Report
- Power BI Dashboard

# 📌 Key Features

- End-to-End Data Analytics Pipeline
- Automated ETL Process
- Feature Engineering
- Machine Learning Forecasting
- Inventory Intelligence
- Power BI Dashboard
- Business Insights
- Retail Analytics
