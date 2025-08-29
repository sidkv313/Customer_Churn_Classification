https://customerchurnclassification-qrb3vyxecmc9y3cmmqhz33.streamlit.app/


# Customer Churn Prediction for Brazilian E-Commerce Platform

![Churn Analysis](https://github.com/sidkv313/Customer_Churn_Classification/edit/main/media/churn_analysis_illustration.jpg)  

---

## Project Overview

This project tackles the critical business problem of **customer churn** in a Brazilian e-commerce platform by leveraging a rich, multi-faceted dataset. The aim is to understand churn drivers through deep data analysis and build a predictive machine learning model that classifies customers likely to churn.

A unique aspect of this work is the heavy reliance on **SQL** to integrate and engineer features from multiple relational data sources, creating a comprehensive view of customer behavior over time.

---

## Technologies and Tools Used

- **SQL (MySQL/SQLite):**  
  Core to this project is the use of advanced SQL to query, join, filter, and aggregate across multiple tables including customers, orders, order items, payments, and reviews.  
  Complex queries are written with joins, aggregates, window functions, and conditional logic to create an Analytics Base Table (ABT) for churn analysis.  
  The churn label is derived directly in SQL based on customer inactivity over a 180-day threshold.

- **Python:**  
  Python scripts and Jupyter notebooks employed for:  
  - Connecting to SQL databases and loading data into Pandas DataFrames.  
  - Detailed Exploratory Data Analysis (EDA) with visualizations using Matplotlib and Seaborn.  
  - Extensive feature engineering beyond SQL aggregation where needed.  
  - Model development using Scikit-learn including Random Forest classifiers.

- **Machine Learning:**  
  Classification algorithms were evaluated to predict churn. Models were assessed using metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
  The best-performing model was selected and saved for deployment.

- **Streamlit:**  
  An interactive front-end app was developed to enable users to input customer data and get real-time churn predictions backed by the trained model.

---

## Dataset Description

The project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), consisting of:  
- 100k+ orders from 2016 to 2018 spanning multiple Brazilian marketplaces.  
- Multiple related tables with customer demographics, order transactions, multi-item orders, payment methods, and customer reviews.  
- An anonymized dataset offering granular insights into customer buying patterns and satisfaction levels.

---

## Project Internals

### SQL-Driven Data Engineering

- Created a **master analytics table** aggregating customer-level metrics such as:  
  - Total orders placed  
  - Total and average spend per order  
  - Average review scores  
  - Number of distinct payment methods used  
  - Customer tenure (days between first and last purchase)  

- Defined **churn** as customers with no purchase in the last 180 days from the latest order date in the dataset.

- This approach enabled efficient and scalable feature creation directly in the database layer before moving data into Python.

### Exploratory Data Analysis (EDA)

- Analyzed customer distribution by geography (states in Brazil).  
- Examined spending behavior, review scores, and payment patterns correlating with churn rates.  
- Visualized key feature correlations and distributions to inform feature selection.

### Machine Learning Pipeline

- Data loaded from SQL into Python DataFrames for further cleaning and feature processing.  
- Developed and compared several classification models.  
- Random Forest provided strong performance and interpretability via feature importance analysis.

### Deployment

- Developed a **Streamlit web app** for user-friendly churn risk predictions by entering key customer metrics.

---

## Project Results and Insights

- SQL-based feature engineering proved powerful for turning raw relational data into actionable metrics.  
- Key churn predictors included **customer tenure**, **average review score**, and **spending patterns**.  
- Customers with longer inactivity (>180 days) after last purchase were reliably identified as churned.  
- The Random Forest model achieved favorable performance metrics, enabling confident churn prediction for business use.  
- The Streamlit app offers interactive predictions, helping stakeholders make data-driven retention decisions.

---

## How to Run

1. Set up the SQL database and load the dataset using provided SQL scripts and Python loaders.  
2. Run exploratory SQL queries and create the master analytics table.  
3. Execute the Jupyter notebook for feature engineering, model training, and evaluation.  
4. Launch the Streamlit app to interact with the churn prediction model:

'
pip install -r requirements.txt
streamlit run app.py
'

---

## Files and Structure

| File                      | Description                                       |
|---------------------------|-------------------------------------------------|
| `2_create_master_table.sql` | SQL script to create consolidated customer analytics table with churn flag |
| `Project-Customer-Churn-Prediction.ipynb` | Complete data science pipeline and modeling notebook |
| `app.py`                  | Streamlit web application for real-time churn prediction |
| Dataset CSVs              | Raw input data from Olist, split across tables  |
| `requirements.txt`        | Project dependencies                             |

---

## Visualizations to Include

- Customer geographic distribution (e.g., customers by Brazilian states).  
- Correlation heatmap of churn-related features.  
- Feature importance chart from the Random Forest model.  
- Churn rate variations by key demographic or behavioral segments.

---

## Future Enhancements

- Incorporating customer review text analysis using NLP for sentiment-driven features.  
- Deploying the model as a RESTful API for seamless integration with other systems.  
- Exploring advanced models like XGBoost and deep learning techniques.  
- Conducting detailed customer segmentation for targeted retention strategies.


