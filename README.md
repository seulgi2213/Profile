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


#### Creating Stacked Datframe for Time-Series Analysis 
- **Code2:** [StackDataset](https://github.com/seulgi2213/Profile/blob/main/Stacked%20Time%20Series%20Dataframe.ipynb) <br>
* Data Cleaning
	•	The initial dataset had 64 variables and over 2.4 billion rows, processed with PySpark on GWU’s High Power Computing system.
	•	Key preprocessing steps:
	•	Handling Missing Data: Columns with over 95% null values were removed.
	•	Merging Datasets: The Performance and Origination datasets were merged using LOAN SEQUENCE NUMBER to provide a complete loan lifecycle view.
	•	Calculating ELTV: To fill missing Estimated Loan-to-Value (ELTV) ratios, ELTV was calculated as the ratio of the current unpaid balance to the adjusted housing price, factoring in House Price Index changes since loan origination.

* Variable Selection
	•	Target Variable: Probability of default.
	•	Input Variables:
	•	Static Variables: Origination Credit Score, Original Interest Rate, Property Type, Loan Purpose, etc.
	•	Dynamic Variables: Current Unpaid Balance, Loan Delinquency Status, Loan Age, Estimated Loan-to-Value.
	•	Macroeconomic Variables: Current Interest Rate, Unemployment Rate, Inflation Rate, House Price Index (nationally used for nulls at the state level).

#### XGBoost Modeling in PiML 
- **Code3:** [Modeling](https://github.com/seulgi2213/Profile/blob/main/Modeling%20with%20XGBoost%20in%20PiML.ipynb) <br>
      
### Machine Learning Projects 
<a name="ML"></a>

#### Bikeshare 
####

### NY State Environment Conservation Project 
<a name="SQL"></a>

### Tableau 
<a name="Tableau"></a>

### Decision Model 
<a name="DM"></a>

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
