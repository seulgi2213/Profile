# Seulgi Ko - Business/Data Analysis Portfolio 

# About me 
Hello! I’m Seulgi Ko, an accounting and finance professional with experience as a consultant at accounting Big 4's. I have worked with clients from various industries, including large banks, cryto, electronics, and e-commerce, providing insights and solutions tailored to their unique needs. Leveraging this business knowledge alongside technical skills from my MS in Business Analytics at George Washington University, I am equipped to drive data-driven solutions in finance/accounting and analytics.

On this dashboard, you can explore projects and coursework that showcase my expertise in analytical skills, including modeling and machine learning, along with technical proficiency in PySpark, SQL, and Python.

---

# Table Content 
* Projects
   * [Wells Fargo Interpretable Machine Learning for Mortgage Default Prediction](#Default)
   * [NY State Environment Conservation Project](#SQL)
   * Machine Learning Projects <br>
       [Bikeshare Optimization](#ML1) <br>
       [Customer Churn Prediction ](#ML2) <br> 
* [Skills/License](#Skill)
* [Contacts](#Contact)

---

# Projects

## Wells Fargo Interpretable Machine Learning for Mortgage Default Prediction <a name="Default"></a>


### Overview
The mortgage market plays a crucial role in financial markets. This project develops a predictive model for mortgage default over the next 24 months, focusing on interpretability, accuracy, and risk management. Key objectives include:
- Enhancing stability of mortgage-backed securities.
- Adapting to diverse economic conditions, including crisis scenarios like COVID-19.
- Integrating static, dynamic, and macroeconomic variables for robust predictions.



### Data

- **Source**: Freddie Mac single-family home loans, from 2000 to Q2 2023.
- **Content**: Origination and performance data, focusing on 30-year fixed-rate mortgages.
- **Macroeconomic Variables**: Housing price index, inflation rate, and unemployment rate.



### Preprocessing and Sampling Techniques

**1. Code**: [Preprocessing Jupyter](https://github.com/seulgi2213/Profile/blob/main/Preprocessing_PySpark.ipynb)

**2. Data Cleaning**
   - **Scope**: The initial dataset had 64 variables and over 2.4 billion rows, processed with PySpark on GWU’s High Power Computing system.
   - **Key Steps**:
     - **Handling Missing Data**: Removed columns with over 95% null values.
     - **Merging Datasets**: Combined Performance and Origination datasets on `LOAN SEQUENCE NUMBER` to create a complete loan lifecycle view.
     - **ELTV Calculation**: Estimated Loan-to-Value (ELTV) ratio calculated as the ratio of current unpaid balance to adjusted housing price, accounting for housing price index changes.

**3. Variable Selection**
   - **Target Variable**: Probability of default.
   - **Input Variables**:
     - **Static Variables**: Origination Credit Score, Original Interest Rate, Property Type, Loan Purpose, etc.
     - **Dynamic Variables**: Current Unpaid Balance, Loan Delinquency Status, Loan Age, ELTV.
     - **Macroeconomic Variables**: Current Interest Rate, Unemployment Rate, Inflation Rate, House Price Index (nationally used for null values at the state level).

**4. Sampling Techniques**
   - **Sampling Strategy**: Selected loans based on `CURRENT LOAN DELINQUENCY STATUS`, with 3,000 loans per year, including 350 defaults and 350 non-defaults per quarter.
   - **Limitations**: Fewer defaults in recent periods (e.g., 264 defaults in Q4 2022 and 32 in Q1 2023).
   - **Additional Time Variables**: Added `OrigDate`, `OrigYear`, and `OrigQuarter` to track quarterly effects.



### Creating the Stacked Dataset for Time-Series Analysis

**1. Code**: [StackDataset Jupyter](https://github.com/seulgi2213/Profile/blob/main/Stacked%20Time%20Series%20Dataframe.ipynb)

**2. Overview of Time Series Horizon**
   - The predictive loan default model uses a paneled time series approach to forecast loan defaults. Each horizon is based on historical data up to a snapshot time (s), with forecasts for future times (t). 
   - To forecast over 24 months, each loan’s last row is duplicated 24 times. This approach improves long-term forecasting by reducing over-reliance on initial data points.

**3. Stacked Dataset Creation Process**
   - **Sampling**: Begins with 3,000 loans per year over 24 years, then converted into a time series format.
   - **Minimum Loan Age**: Identified for each loan as the starting point.
   - **Sequential Loan Age Adjustment**: Ensures chronological loan age progression within each horizon.
   - **Replication**: Each row duplicated 24 times to simulate a 24-month forecast period.

**4. Additional Columns**
   - **HORIZON**: Tracks future periods for each duplicated row.
   - **SOURCE**: Differentiates original rows (“orig”) from generated rows for forecasting (“Duplicated”).

**5. Example of Stacked Data**

| Group | DEFAULT | Horizon | Source | LOAN SEQUENCE NUMBER | MONTHLY REPORTING PERIOD | CURRENT ACTUAL UPB | CURRENT LOAN DELINQUENCY STATUS | LOAN AGE | CURRENT INTEREST RATE |
|-------|---------|---------|--------|-----------------------|---------------------------|---------------------|---------------------------------|----------|------------------------|
| 0     | 0       | 0       | orig   | F00Q10000066         | 2000-02                   | 132000.0           | 0                               | 0        | 8.0                    |
| 1     | 0       | 0       | orig   | F00Q10000066         | 2000-03                   | 132000.0           | 0                               | 1        | 8.0                    |
| 1     | 0       | 1       | Dupli… | F00Q10000066         | 2000-02                   | 132000.0           | 0                               | 0        | 8.0                    |
| 2     | 0       | 0       | orig   | F00Q10000066         | 2000-04                   | 131000.0           | 0                               | 2        | 8.0                    |
| 2     | 0       | 1       | Dupli… | F00Q10000066         | 2000-03                   | 132000.0           | 0                               | 1        | 8.0                    |



### XGBoost Modeling in PiML

**1. Code**: [Modeling Jupyter](https://github.com/seulgi2213/Profile/blob/main/Modeling%20with%20XGBoost%20in%20PiML.ipynb)

**2. XGBoost Results**

| Model   | test_ACC | test_AUC | test_F1 | test_LogLoss | test_Brier | train_ACC | train_AUC | train_F1 | train_LogLoss | train_Brier |
|---------|----------|----------|---------|--------------|------------|-----------|-----------|----------|---------------|-------------|
| XGB2    | 0.6656   | 0.7361   | 0.6367  | 0.6137       | 0.2125     | 0.7004    | 0.7695    | 0.7083   | 0.5729        | 0.1952      |
| XGB2_v2 | 0.6681   | 0.7287   | 0.6416  | 0.6257       | 0.2160     | 0.7003    | 0.7702    | 0.7070   | 0.5726        | 0.1950      |

**3. Model Accuracy: XGB2_v2**

|         | ACC   | AUC   | F1    | LogLoss | Brier |
|---------|-------|-------|-------|---------|-------|
| Train   | 0.7003| 0.7702| 0.7070| 0.5726  | 0.1950|
| Test    | 0.6681| 0.7287| 0.6414| 0.6257  | 0.2160|
| Gap     |-0.0323|-0.0415|-0.0656| 0.0530  | 0.0210|



### Visualizations

- **Residual Box Plot of Predicted Default Variable (XGB2_v2 Model)**
  
  <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/9e8f2fcd-ef9f-4825-8d33-c5effd0d0bf7" alt="Residual Box Plot" width="600">

- **Predicted vs. Actual Default Rate: XGB2_v2**
  
  <img src="https://github.com/celinawong21/WF-ML-Model/assets/158225115/c6e6ba99-25e1-4794-bb0f-2e6d6496f522" alt="Predicted vs. Actual" width="700">







# Bikeshare Optimization <a name="ML1"></a>



### Purpose
The objective of this project is to enhance bikeshare operation efficiency by accurately predicting pickup and dropoff demand. Precise forecasting is crucial for optimal resource allocation, minimizing unmet demand, and maximizing profitability. This study investigates five regression models to identify the best approach for predicting pickups and dropoffs:
  - **Linear Regression**
  - **Lasso Regression**
  - **Ridge Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Elastic Net Regression**

Additionally, two operational strategies were considered:
  - **Cost Strategy**: Reducing costs associated with unsuccessful pickups and dropoffs due to demand misalignment.
  - **Quality Strategy**: Ensuring a high quality of service by aligning predictions with actual demand to reduce wait times and improve user satisfaction.



### Model Development

#### 1. Exploratory Data Analysis (EDA)

- **Data Integration**:  
  Bikeshare usage data (pickup and dropoff counts) was combined with weather data, providing insights into how weather conditions affect bikeshare demand.

- **Feature Selection**:  
  A correlation matrix was used to analyze relationships between weather features and bikeshare demand, leading to the selection of five key features:
  - `tempmax` (maximum temperature)
  - `feelslikemax` (perceived temperature)
  - `precipcover` (precipitation coverage)
  - `humidity`
  - `windspeed`

- **Data Cleaning**:  
  Irrelevant or redundant columns were removed to focus on impactful features, ensuring that the data was clean and suitable for model training.

#### 2. Predictive Modeling: Model Setup and Training

Each model was trained to predict pickup and dropoff counts using the selected weather features as inputs. Below is an overview of each model used:

- **Linear Regression**:  
  A straightforward linear approach for predicting demand based on feature relationships.

- **Lasso Regression**:  
  Linear model with L1 regularization, which shrinks less important features toward zero, effectively performing feature selection.

- **Ridge Regression**:  
  Linear model with L2 regularization, controlling for multicollinearity by reducing the impact of correlated features.

- **Elastic Net Regression**:  
  A combination of L1 and L2 regularization (Lasso and Ridge), effective when handling sparse data with correlated features.

- **K-Nearest Neighbors (KNN)**:  
  A non-parametric model that predicts pickups/dropoffs based on similar past data points, suitable for capturing non-linear patterns.

#### Evaluation Metrics
Each model’s performance was evaluated based on:

- **R-Squared (R²)**: Measures the proportion of variance in demand that each model explains, with higher values indicating better fit.
- **Mean Squared Error (MSE)**: Measures average squared differences between predicted and actual values, where lower values reflect more accurate predictions.



### Results

The results focused on prediction accuracy (R² and MSE) and operational metrics (cost and quality of service) for each model:

- **Prediction Accuracy Results**:
  - **Pickup Accuracy**:  
    Linear Regression had the highest R-squared value of **0.34**, making it the most effective model for predicting pickups. With an MSE of **83.054**, it demonstrated a good balance between explaining variability in pickups and minimizing prediction error.

  - **Dropoff Accuracy**:  
    Lasso Regression outperformed other models, achieving the highest R-squared value of **0.344** for dropoffs. Its lower MSE of **74.86** further indicated that it was the best model for accurately predicting dropoffs.

This analysis led to the recommendation of **Linear Regression for pickup accuracy** and **Lasso Regression for dropoff accuracy**.

- **Cost Strategy**:  
  The cost strategy aimed to minimize financial impacts from unmet pickup and dropoff demand. Each failed pickup was assigned a cost of $2, and each failed dropoff a cost of $3. By comparing predicted and actual values, the models’ costs of unmet demand were calculated.
  - **Results**:  
    Linear Regression showed the lowest average total cost for unsuccessful predictions, making it the optimal model for minimizing operational costs.

- **Quality Strategy**:  
  Quality of Service (QoS) was measured based on the alignment between model predictions and actual demand, focusing on reducing unmet demand and providing reliable service.
  - **Results**:  
    Linear Regression achieved the highest average QoS score, indicating superior performance in meeting demand accurately and consistently.



### Skills Earned:
  - [x] 📊 **Data Analysis and Preprocessing**
  - [x] 🧠 **Statistical Modeling and Machine Learning**
  - [x] 💻 **Python Programming** (Pandas, NumPy, Scikit-learn)
  - [x] 📈 **Data Visualization** (Matplotlib, Seaborn)
  - [x] 📐 **Regression and Model Evaluation Metrics**







## Customer Churn Prediction <a name="ML2"></a>



### Purpose
The objective of this project is to improve customer retention for a bank by predicting customer churn using machine learning models. By proactively identifying at-risk customers, the bank can implement targeted retention strategies to minimize revenue loss and enhance profitability. This study investigates the effectiveness of four machine learning models:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **XGBoost**

Two primary outcomes were considered:
  - **Revenue Impact**: Estimating the potential revenue loss from churn and the revenue gain from retention.
  - **Retention Strategy**: Providing insights into key factors driving churn to support personalized retention efforts.

    

### Dataset Summary

**Data Source**: The dataset, sourced from Kaggle, includes 10,000 observations and 12 features on customer data. The target variable, **Exited**, indicates if a customer left the bank.

- **Key Variables**:
    - **CreditScore**: Higher scores may correlate with lower churn risk.
    - **Geography**: Customer location could impact retention.
    - **Gender**: Gender might affect customer churn behavior.
    - **Age**: Older customers tend to have higher retention rates.
    - **Tenure**: Long-tenure customers are often more loyal.
    - **Balance**: High balances correlate with lower churn risk.
    - **NumOfProducts**: Number of bank products held by the customer.
    - **HasCrCard**: Indicates if the customer has a credit card.
    - **IsActiveMember**: Active customers are less likely to leave.
    - **EstimatedSalary**: Lower salaries might indicate higher churn risk.



### Data Preprocessing

- **Data Cleaning**: The dataset contained no missing values, ensuring reliable data inputs.
- **Feature Engineering**: Categorical variables (e.g., *Geography*, *Gender*) were converted to numerical values.
- **Correlation Analysis**: A correlation heatmap highlighted *Age* and *Balance* as influential in predicting churn. Features with low correlation to churn, such as *CustomerId* and *Geography*, were excluded from modeling.



### Modeling

We evaluated the following models based on **precision**, **recall**, and **F1-score** metrics to classify churn (Class 1) vs. non-churn (Class 0):

  - **Logistic Regression**: Performed well in identifying non-churn cases (F1-score 0.90), but struggled with churn cases (F1-score 0.47).
  - **Decision Tree**: Balanced results for non-churn (F1-score 0.86) but low performance for churn detection (F1-score 0.45).
  - **Random Forest**: Demonstrated the best performance for non-churn (F1-score 0.92) and the highest F1-score for churn (0.56).
  - **XGBoost**: Competitive with Random Forest, showing robust F1-scores for both non-churn (0.91) and churn (0.53).



### Revenue Prediction Assumptions
To estimate the revenue impact of customer churn predictions, the following assumptions were applied:
  - **Investment Return**: A projected quarterly return of 15% on invested customer funds.
  - **Interest Rate on Savings**: A standard annual interest rate of 3% on customer balances.
  - **Retention Incentive**: For customers predicted to churn, the interest rate is increased to 5% to encourage retention.

These assumptions reflect typical banking practices and provide a basis for comparing revenue before and after implementing the predictive model.



### Model Performance

| Model           | Class | Precision | Recall | F1-Score |
|-----------------|-------|-----------|--------|----------|
| Logistic        | 0     | 0.86      | 0.95   | 0.90     |
|                 | 1     | 0.65      | 0.37   | 0.47     |
| Decision Tree   | 0     | 0.86      | 0.86   | 0.86     |
|                 | 1     | 0.45      | 0.45   | 0.45     |
| Random Forest   | 0     | 0.88      | 0.96   | 0.92     |
|                 | 1     | 0.73      | 0.45   | 0.56     |
| XGBoost         | 0     | 0.87      | 0.95   | 0.91     |
|                 | 1     | 0.67      | 0.44   | 0.53     |

**Summary**: **Random Forest** and **XGBoost** emerged as the top models, with Random Forest slightly outperforming in terms of F1-scores, especially for non-churn cases.



### ROC-AUC Analysis

ROC-AUC values further evaluated each model’s ability to distinguish churn from non-churn.

| Model           | AUC (OVO) | AUC (OVR) |
|-----------------|-----------|-----------|
| Logistic        | 0.8253    | 0.8253    |
| Decision Tree   | 0.6625    | 0.6625    |
| Random Forest   | 0.8468    | 0.8468    |
| XGBoost         | 0.8364    | 0.8364    |

**Conclusion**: **Random Forest** and **XGBoost** consistently demonstrated high AUC values, making them the most reliable models for churn prediction.



### Decision Performance Evaluation

#### Revenue Projection

- **Pre-Model Revenue**: The projected quarterly revenue was approximately **$11.43 million**.
- **Post-Model Revenue**: After employing targeted retention strategies based on model predictions, the revenue increased to **$13.42 million**, demonstrating a **$1.98 million** gain.

#### Revenue Breakdown

- **True Positives (Churn Predicted Correctly)**: Revenue generated from retaining at-risk customers through incentives.
- **False Positives (Non-Churn Predicted as Churn)**: Represents additional costs for incentives that were not needed.
- **Non-Churned Customers**: Successful retention led to consistent revenue.
- **Churned Customers**: Revenue loss from customers who left, which could be mitigated with improved prediction accuracy.

#### Total Revenue Impact

The machine learning model increased total quarterly revenue by approximately **$1.98 million**, highlighting the financial value of data-driven churn prediction and retention strategies.



### Conclusion
The XGBoost model proved effective in optimizing the bank’s customer retention strategy, leading to a **$1.98 million** revenue increase per quarter. This outcome emphasizes the importance of machine learning in identifying at-risk customers and implementing targeted retention incentives.



### Recommendations
- **Incentive Plans**: Offer tailored incentives, such as higher interest rates or fee waivers, for at-risk customers.
- **Communication Strategy**: Use multi-channel outreach (mail, email, SMS) based on customer demographics for maximum engagement.
- **Referral Program**: Encourage existing customers to refer friends, offering a bonus for new accounts meeting minimum balance requirements.



### Skills Earned:
  - [x] 📊 **Data Analysis and Preprocessing**
  - [x] 🧠 **Statistical Modeling and Machine Learning**
  - [x] 💻 **Python Programming** (Pandas, NumPy, Scikit-learn)
  - [x] 📈 **Data Visualization** (Matplotlib, Seaborn)
  - [x] 📐 **Classification Metrics and Model Evaluation**







## NY State Environment Conservation Project <a name="SQL"></a>





---

## Skills/License 
<a name="Skill"></a>

* Programming Languages & Tools: Python(Pandas, NumPy, MatPlotLib, Scikit-learn), SQL, PySpark, AWS
* Statistical Analysis & Data Processing: R, MS Excel (Solver, Pivot Table, VLOOKUP)
* PUT Decison Model I learned 
* License: CPA, Washington (License No: 40792) — Expiration Date: 06/30/2026

---

## Contacts 
<a name="Contact"></a>
* Linkedin : [Seulgi Ko](http://www.linkedin.com/in/seulgi-ko)
* Email: seulgi2213@gmail.com
