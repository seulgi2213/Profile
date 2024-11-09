# Seulgi Ko - Business/Data Analysis Portfolio

## About Me 
Hello! I’m Seulgi Ko, an accounting and finance professional with experience as a consultant at Big 4 accounting firms. I have worked with clients across industries, including large banks, crypto, electronics, and e-commerce, providing insights and solutions tailored to their unique needs. Leveraging my business knowledge alongside technical skills from my MS in Business Analytics at George Washington University, I am equipped to drive data-driven solutions in finance, accounting, and analytics.

On this dashboard, you can explore projects and coursework that showcase my expertise in analytical skills, including modeling and machine learning, along with technical proficiency in PySpark, SQL, and Python.

---

## Table of Contents
- **Projects**
  - [Wells Fargo Interpretable Machine Learning for Mortgage Default Prediction](#Default)
  - [NY State Environment Conservation Project](#SQL)
  - Machine Learning Projects  
    - [Bikeshare Optimization](#ML1)  
    - [Customer Churn Prediction](#ML2)  
- [Skills and Licenses](#Skill)
- [Contact Information](#Contact)

---

## Projects

### Wells Fargo Interpretable Machine Learning for Mortgage Default Prediction <a name="Default"></a>

- **Overview**:  
  The mortgage market plays a crucial role in financial markets. This project develops a predictive model for mortgage default over the next 24 months, focusing on interpretability, accuracy, and risk management. Key objectives include:
  - Enhancing the stability of mortgage-backed securities.
  - Adapting to diverse economic conditions, including crises like COVID-19.
  - Integrating static, dynamic, and macroeconomic variables for robust predictions.

- **Data**:
  - **Source**: Freddie Mac single-family home loans, from 2000 to Q2 2023.
  - **Content**: Origination and performance data for 30-year fixed-rate mortgages.
  - **Macroeconomic Variables**: Housing price index, inflation rate, and unemployment rate.

- **Preprocessing and Sampling Techniques**:
  - [Code: Preprocessing Jupyter Notebook](https://github.com/seulgi2213/Profile/blob/main/Preprocessing_PySpark.ipynb)
  - **Data Cleaning**:
    - Scope: 64 variables and over 2.4 billion rows, processed with PySpark on GWU’s High Power Computing system.
    - Key Steps:
      - Handling Missing Data: Removed columns with over 95% null values.
      - Merging Datasets: Combined Performance and Origination datasets on `LOAN SEQUENCE NUMBER` to create a complete loan lifecycle view.
      - ELTV Calculation: Estimated Loan-to-Value (ELTV) ratio calculated as the ratio of current unpaid balance to adjusted housing price, accounting for housing price index changes.
  - **Variable Selection**:
    - Target Variable: Probability of default.
    - Input Variables:  
      - Static Variables: Origination Credit Score, Original Interest Rate, Property Type, Loan Purpose, etc.
      - Dynamic Variables: Current Unpaid Balance, Loan Delinquency Status, Loan Age, ELTV.
      - Macroeconomic Variables: Current Interest Rate, Unemployment Rate, Inflation Rate, House Price Index (nationally used for null values at the state level).
  - **Sampling Strategy**: Selected loans based on `CURRENT LOAN DELINQUENCY STATUS`, with 3,000 loans per year, including 350 defaults and 350 non-defaults per quarter.

- **Creating the Stacked Dataset for Time-Series Analysis**:
  - [Code: Stacked Dataset Jupyter Notebook](https://github.com/seulgi2213/Profile/blob/main/Stacked%20Time%20Series%20Dataframe.ipynb)
  - Overview of Time Series Horizon: The predictive loan default model uses a paneled time series approach, where each loan’s last row is duplicated 24 times for long-term forecasting.

- **XGBoost Modeling in PiML**:
  - [Code: Modeling Jupyter Notebook](https://github.com/seulgi2213/Profile/blob/main/Modeling%20with%20XGBoost%20in%20PiML.ipynb)
  - **Model Accuracy**: Key metrics for the XGB2_v2 model include test AUC of 0.7287 and test F1-score of 0.6416.

- **Visualizations**:
  - Residual Box Plot and Predicted vs. Actual Default Rate.

---

### Bikeshare Optimization <a name="ML1"></a>

- **Purpose**:  
  This project aims to enhance bikeshare operation efficiency by accurately predicting pickup and dropoff demand, improving resource allocation, reducing unmet demand, and maximizing profitability. Models evaluated include:
  - **Linear Regression**
  - **Lasso Regression**
  - **Ridge Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Elastic Net Regression**

- **Model Development**:
  - **Exploratory Data Analysis (EDA)**:
    - Data Integration: Combined bikeshare usage data with weather data to understand how weather affects bikeshare demand.
    - Feature Selection: Selected variables including `tempmax`, `feelslikemax`, `precipcover`, `humidity`, and `windspeed`.
  - **Predictive Modeling**: Models trained to predict pickup and dropoff counts using the selected weather features.
  - **Evaluation Metrics**: R-Squared and Mean Squared Error (MSE) were used to evaluate each model's performance.

- **Results**:
  - **Pickup Accuracy**: Linear Regression achieved the highest R-squared of 0.34.
  - **Dropoff Accuracy**: Lasso Regression achieved the highest R-squared of 0.344.
  - **Cost and Quality Strategies**:
    - Cost Strategy: Linear Regression minimized the cost of unmet demand.
    - Quality Strategy: Linear Regression provided the highest quality of service alignment.

- **Skills Earned**:
  - 📊 Data Analysis and Preprocessing
  - 🧠 Statistical Modeling and Machine Learning
  - 💻 Python Programming (Pandas, NumPy, Scikit-learn)
  - 📈 Data Visualization (Matplotlib, Seaborn)
  - 📐 Regression and Model Evaluation Metrics

---

### Customer Churn Prediction <a name="ML2"></a>

- **Purpose**:  
  This project improves customer retention for a bank by predicting customer churn, enabling proactive, targeted retention strategies to reduce revenue loss.

- **Dataset Summary**:
  - **Source**: Kaggle, with 10,000 observations and 12 features on customer data.
  - Key Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

- **Data Preprocessing**:
  - **Data Cleaning**: No missing values were found.
  - **Feature Engineering**: Converted categorical variables to numerical, selected key features for model training.

- **Modeling**:
  - **Models Evaluated**:
    - Logistic Regression, Decision Tree, Random Forest, XGBoost.
  - **Revenue Prediction Assumptions**:
    - Investment Return: 15% quarterly return on invested customer funds.
    - Interest Rate on Savings: 3% annual interest, increased to 5% for at-risk customers.

- **Results**:
  - **Model Performance**: Random Forest and XGBoost emerged as the top models.
  - **Decision Performance Evaluation**:
    - Pre-Model Revenue: $11.43 million.
    - Post-Model Revenue: $13.42 million, a $1.98 million gain.
  - **Conclusion**: XGBoost optimized retention, increasing quarterly revenue by $1.98 million.

- **Skills Earned**:
  - 📊 Data Analysis and Preprocessing
  - 🧠 Statistical Modeling and Machine Learning
  - 💻 Python Programming (Pandas, NumPy, Scikit-learn)
  - 📈 Data Visualization (Matplotlib, Seaborn)
  - 📐 Classification Metrics and Model Evaluation

---

### NY State Environment Conservation Project <a name="SQL"></a>

*Project details will go here once completed.*

---

## Skills and Licenses <a name="Skill"></a>

- **Programming Languages & Tools**:
  - Python (Pandas, NumPy, MatPlotLib, Scikit-learn), SQL, PySpark, AWS
- **Statistical Analysis & Data Processing**:
  - R, MS Excel (Solver, Pivot Table, VLOOKUP)
- **Decision Modeling**
- **License**:
  - CPA, Washington (License No: 40792) — Expiration Date: 06/30/2026

---

## Contact Information <a name="Contact"></a>

- **LinkedIn**: [Seulgi Ko](http://www.linkedin.com/in/seulgi-ko)
- **Email**: seulgi2213@gmail.com
