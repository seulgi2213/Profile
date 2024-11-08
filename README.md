# Seulgi Ko - Business/Data Analysis Portfolio 

## About me 
Hello! I’m Seulgi Ko, an accounting and finance professional with experience as a consultant at accounting Big 4's. I have worked with clients from various industries, including large banks, cryto, electronics, and e-commerce, providing insights and solutions tailored to their unique needs. Leveraging this business knowledge alongside technical skills from my MS in Business Analytics at George Washington University, I am equipped to drive data-driven solutions in finance/accounting and analytics.

On this dashboard, you can explore projects and coursework that showcase my expertise in analytical skills, including modeling and machine learning, along with technical proficiency in PySpark, SQL, and Python.

## Table Content 
* Projects
   * [Wells Fargo Interpretable Machine Learning for Mortgage Default Prediction](#Default)
   * [NY State Environment Conservation Project](#SQL)
   * [Machine Learning Projects](#ML)
* [Skills/License](#Skill)
* [Contacts](#Contact)

## Projects 
<a name="Default"></a>

### Wells Fargo Interpretable Machine Learning for Mortgage Default Prediction 

#### Overview 
The significance of the mortgage market is immense, highlighting its pivotal role in financial markets. Key business outcomes for mortgage models include achieving interpretability and accuracy to foster trust and understanding among stakeholders, as well as providing default predictions over an extended period for effective risk management. Risk mitigation efforts aim to enhance the stability of mortgage-backed securities, particularly by addressing potential downward trends. Adaptability across diverse economic scenarios, including stress testing during crises like COVID-19, is vital for evaluating the model’s robustness. The objective of this project is to develop a predictive model for mortgage default over the next 24 months. The model will integrate various static, dynamic, and macroeconomic variables to enhance accuracy and robustness.

#### The Data 
* Single-family home loans from Freddie Mac, spanning from 2000 to Q2 2023, consisting of origination and performance data, with a focus on 30-year fixed-rate mortgages.
* Macroeconomic variables, including the housing price index, inflation rate, and unemployment rate.

#### Preprocessing and Sampling Techniques 

- **Code1:** [Preprocessing](https://github.com/seulgi2213/Profile/blob/main/Preprocessing_PySpark.ipynb) <br>
- **Data Cleaning**
  - The initial dataset had 64 variables and over 2.4 billion rows, processed with PySpark on GWU’s High Power Computing system.
  - Key preprocessing steps:
    - **Handling Missing Data**: Columns with over 95% null values were removed.
    - **Merging Datasets**: The Performance and Origination datasets were merged using LOAN SEQUENCE NUMBER to provide a complete loan lifecycle view.
    - **Calculating ELTV**: To fill missing Estimated Loan-to-Value (ELTV) ratios, ELTV was calculated as the ratio of the current unpaid balance to the adjusted housing price, factoring in House Price Index changes since loan origination.

- **Variable Selection**
  - **Target Variable**: Probability of default.
  - **Input Variables**:
    - Static Variables: Origination Credit Score, Original Interest Rate, Property Type, Loan Purpose, etc.
    - Dynamic Variables: Current Unpaid Balance, Loan Delinquency Status, Loan Age, Estimated Loan-to-Value.
    - Macroeconomic Variables: Current Interest Rate, Unemployment Rate, Inflation Rate, House Price Index (nationally used for nulls at the state level).

- **Sampling Techniques**
  - **Methodology**
    - **Sampling Strategy**: Given the dataset’s size, strategic sampling was used based on CURRENT LOAN DELINQUENCY STATUS.
      - **Default Criteria**: Loans with a delinquency status of 6+ months late or marked “RA” are classified as defaults; all others as non-defaults.
      - **True Default**: Loans meeting default criteria at any point are labeled as “true_default” for accurate tracking.
      - **Sampling Proportion**: 3,000 loans were sampled per year, including 350 defaults and 350 non-defaults per quarter to maintain balance.
    - **Limitations**: Fewer defaults were found in some recent periods (e.g., only 264 defaults in Q4 2022, 32 in Q1 2023, and none in Q2 2023).
    - **Time Variables**: Added OrigDate, OrigYear, and OrigQuarter to capture quarterly effects for modeling.


   
#### Creating Stacked Datframe for Time-Series Analysis 
- **Code2:** [StackDataset](https://github.com/seulgi2213/Profile/blob/main/Stacked%20Time%20Series%20Dataframe.ipynb) <br>
- **Overview of Time Series Horizon**
  - Our predictive loan default model uses a paneled time series approach to forecast loan defaults over future periods. Each prediction horizon (future period) is based on all available historical data up to a certain snapshot time (s), where predictions are made for times (t) beyond s. This approach creates pairs of snapshot-forecast data, forming what we refer to as stacked data. To forecast each loan’s default probability over a 24-month period, we duplicate each row (last row) 24 times. This time series horizon approach, chosen over traditional time series modeling, provides more robust long-term predictions by reducing reliance on initial data points.

- **Creating a Stacked Dataset**
  - **Vectorized Process**
    - **Sampling**: We start with 3,000 loans from each year over 24 years, then convert this into a time series format.
	  - **Minimum LOAN AGE**: Each loan’s minimum LOAN AGE is identified as its starting point.
	  - **Chronological Order**: If multiple loans have the same LOAN AGE at a given horizon, we adjust the dataset to ensure each loan’s age progresses sequentially.
	  - **Replication**: Each row is duplicated 24 times to simulate a 24-month forecast period, allowing for extended loan behavior predictions.

  - **Additional Columns for Analysis**
	  -	**HORIZON**: Tracks historical data for each duplicated row, with each horizon representing one month prior to the snapshot.
	  - **SOURCE**: Distinguishes between original rows (“orig”) and generated rows for forecasting (“Duplicated”).
    - 
- **Example of Stacked Data**
| Group | DEFAULT | Horizon | Source | LOAN SEQUENCE NUMBER | MONTHLY REPORTING PERIOD | CURRENT ACTUAL UPB | CURRENT LOAN DELINQUENCY STATUS | LOAN AGE | CURRENT INTEREST RATE |
|-------|---------|---------|--------|-----------------------|---------------------------|---------------------|---------------------------------|----------|-----------------------|
| 0     | 0       | 0       | orig   | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |
| 1     | 0       | 0       | orig   | F00Q10000066          | 2000-03                   | 132000.0            | 0                               | 1        | 8.0                   |
| 1     | 0       | 1       | Dupli… | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |
| 2     | 0       | 0       | orig   | F00Q10000066          | 2000-04                   | 131000.0            | 0                               | 2        | 8.0                   |
| 2     | 0       | 1       | Dupli… | F00Q10000066          | 2000-03                   | 132000.0            | 0                               | 1        | 8.0                   |
| 2     | 0       | 2       | Dupli… | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |
| 3     | 0       | 0       | orig   | F00Q10000066          | 2000-05                   | 131000.0            | 0                               | 3        | 8.0                   |
| 3     | 0       | 1       | Dupli… | F00Q10000066          | 2000-04                   | 131000.0            | 0                               | 2        | 8.0                   |
| 3     | 0       | 2       | Dupli… | F00Q10000066          | 2000-03                   | 132000.0            | 0                               | 1        | 8.0                   |
| 3     | 0       | 3       | Dupli… | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |


#### XGBoost Modeling in PiML 
- **Code3:** [Modeling](https://github.com/seulgi2213/Profile/blob/main/Modeling%20with%20XGBoost%20in%20PiML.ipynb) <br>

- **XGBoost2 Result**
  
|     | Model    | test_ACC | test_AUC | test_F1 | test_LogLoss | test_Brier | train_ACC | train_AUC | train_F1 | train_LogLoss | train_Brier |
|-----|----------|----------|----------|---------|--------------|------------|-----------|-----------|----------|---------------|-------------|
|  0  | XGB2     | 0.6656   | 0.7361   | 0.6367  | 0.6137       | 0.2125     | 0.7004    | 0.7695    | 0.7083   | 0.5729        | 0.1952      |
|  1  | XGB2_v2  | 0.6681   | 0.7287   | 0.6416  | 0.6257       | 0.2160     | 0.7003    | 0.7702    | 0.7070   | 0.5726        | 0.1950      |


- **Accuracy Descriptions of XGB2_v2 Model**
|         |  ACC   |  AUC   |   F1   | LogLoss |  Brier |
|---------|--------|--------|--------|---------|--------|
|  Train  | 0.7003 | 0.7702 | 0.7070 |  0.5726 | 0.1950 |
|  Test   | 0.6681 | 0.7287 | 0.6414 |  0.6257 | 0.2160 |
|   Gap   |-0.0323 |-0.0415 |-0.0656 |  0.0530 | 0.0210 |

<img width="500" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/d684fd69-ea70-461c-a996-7fd36acbe9e9">

- **Residual Box Plot of Predicted Default Variable from XGB2_v2 Model**

<img width= "650" alt = "image" src= "https://github.com/celinawong21/WF-ML-Model/assets/159848729/9e8f2fcd-ef9f-4825-8d33-c5effd0d0bf7">

- **Predicted vs. Actual Default Rate: XGB2_v2**

<img width="600" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/158225115/c6e6ba99-25e1-4794-bb0f-2e6d6496f522">

### Machine Learning Projects 
<a name="ML"></a>

#### Bikeshare 
####

### NY State Environment Conservation Project 
<a name="SQL"></a>

### Tableau 
<a name="Tableau"></a>

## Skills/License 
<a name="Skill"></a>

* Programming Languages & Tools: Python(Pandas, NumPy, MatPlotLib, Scikit-learn), SQL, PySpark, AWS
* Statistical Analysis & Data Processing: R, MS Excel (Solver, Pivot Table, VLOOKUP)
* PUT Decison Model I learned 
* License: CPA, Washington (License No: 40792) — Expiration Date: 06/30/2026

## Contacts 
<a name="Contact"></a>
* Linkedin : [Seulgi Ko](http://www.linkedin.com/in/seulgi-ko)
* Email: seulgi2213@gmail.com
