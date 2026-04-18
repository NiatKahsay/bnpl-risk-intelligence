# BNPL Credit Risk Prediction Using Machine Learning

This project is a part of the ADS-599 Capstone course in the Applied Data Science Program at the University of San Diego.

**Project Status: Completed**

---

## Installation

To run this project on your machine, follow the steps below.

**1. Clone the repository**
```bash
git clone https://github.com/NiatKahsay/bnpl-risk-intelligence.git
cd bnpl-risk-intelligence
```

**2. Install required Python packages**
```bash
pip install pandas numpy scikit-learn xgboost shap groq fredapi matplotlib seaborn tqdm pyarrow
```

**3. Set up Google Colab (recommended)**

Upload the notebooks to Google Colab and mount your Google Drive. The project folder structure will be created automatically when you run the data collection notebook. Place the raw Lending Club dataset (`accepted_2007_to_2018Q4.csv`) and the CFPB complaints file (`complaints.csv`) in the following path on your Drive:

```
MyDrive/BNPL_Capstone/data/raw/
```

**4. API keys required**
- FRED API: obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html
- Groq API: obtain a free key at https://console.groq.com

Replace the placeholder keys in the data collection and modeling notebooks before running.

**5. Run notebooks in order**
```
01_BNPL_Data_Collection.ipynb
02_BNPL_EDA_Feature_Engineering.ipynb
03_BNPL_Modeling_and_Evaluation_llm.ipynb
04_BNPL_UserArtifact.ipynb
```

---

## Project Intro / Objective

The rapid expansion of buy now, pay later lending has created a gap in credit risk management. BNPL platforms extend unsecured credit with limited underwriting infrastructure, making it difficult to identify high-risk borrowers before credit is issued. This project builds a machine learning pipeline that predicts borrower default risk using historical loan data combined with macroeconomic indicators and consumer complaint signals.

The goal is to demonstrate that a predictive model trained on publicly available data can provide meaningful risk screening for BNPL-like lending environments. The final system produces a default probability for each borrower, explains the prediction using SHAP values, generates a plain-language summary through a large language model, and allows lenders to adjust the classification threshold based on their risk tolerance. The model is deployed through an interactive Streamlit dashboard that enables users to input borrower data, explore different economic scenarios, and view real-time risk predictions: https://bnpl-risk-intelligence.streamlit.app/.

---

## Partners / Contributors

- Niyat Kahsay
- Saloni Bahte
- Kiara Paz

University of San Diego, Applied Data Science Program, Spring 2026

---

## Methods Used

- Binary classification
- Predictive modeling
- Ensemble methods
- Hyperparameter tuning
- Feature engineering
- External data integration
- Model interpretability
- Threshold analysis
- Data visualization
- Machine learning

---

## Technologies

- Python
- Google Colab
- XGBoost
- scikit-learn
- SHAP
- Groq API (Llama 3.1)
- FRED API
- Pandas, NumPy, Matplotlib, Seaborn
- Streamlit (dashboard)
- GitHub

---

## Project Description

**Dataset**

The project combines three data sources into a single loan-level modeling dataset.

The primary dataset is the LendingClub loan dataset (2007 to 2018), sourced from Kaggle. It contains 1,348,099 loan records with borrower characteristics including credit grade, income, debt-to-income ratio, interest rate, FICO score, and loan purpose. The binary target variable was derived from loan status: loans marked as Charged Off or Default were labeled 1, and loans marked as Fully Paid were labeled 0. The resulting default rate is approximately 20%.

The second source is the FRED macroeconomic database, which provided eight monthly economic indicators including credit card delinquency rate, personal savings rate, unemployment rate, consumer sentiment index, and CPI inflation proxy, covering January 2018 onward.

The third source is the CFPB Consumer Complaint Database, filtered to complaints filed against BNPL-related companies including Affirm, Klarna, Afterpay, Zip, Sezzle, and PayPal. Complaint counts were aggregated at the national monthly level and at the state-month level to create behavioral context features. After merging all three sources, the final modeling dataset contains 1,348,099 rows and 40 columns.

**Questions and Hypotheses**

- Can machine learning models trained on historical loan data reliably predict BNPL default risk?
- Do macroeconomic conditions at the time of loan issuance improve prediction beyond borrower-level features alone?
- Does consumer complaint volume serve as a useful behavioral proxy for financial stress at the macro level?
- How does classification threshold selection affect the tradeoff between approval volume and default exposure?

**Modeling**

Three models were trained and evaluated on a stratified 80/20 train/test split. Logistic Regression served as the interpretable baseline and was trained on a 200,000-row sample. Random Forest was trained on 300,000 rows with and without hyperparameter tuning using RandomizedSearchCV. XGBoost was selected as the final model based on its highest test AUC-ROC of 0.7261 and strongest recall on the default class. Class imbalance was addressed using class_weight='balanced' for Logistic Regression and Random Forest, and scale_pos_weight for XGBoost. All models were evaluated using AUC-ROC, precision, recall, F1-score, and the train-to-test AUC gap as an overfitting check.

**Interpretability**

SHAP values were computed on a 500-record test sample using XGBoost's TreeExplainer. The top drivers of default risk were interest rate, loan term, CPI inflation proxy, FICO score, and CFPB complaint count. A Groq API integration using the Llama 3.1 language model converts each borrower's SHAP-based risk factors into a plain-language explanation for non-technical stakeholders.

**Challenges**

The main technical challenges included data leakage from loan status-derived features that produced perfect AUC scores in early runs, memory constraints in Google Colab that required strategic sampling throughout the pipeline, and class imbalance that made accuracy a misleading primary metric. Resolving leakage, switching to targeted imputation, and adopting AUC-ROC as the primary metric were the key fixes that stabilized the pipeline.

---

## License

This project is licensed under the MIT License. See the LICENSE file in the repository for details.

---

## Acknowledgments

The team thanks the instructors and faculty of the University of San Diego Applied Data Science Program for their guidance throughout this project. We also acknowledge the LendingClub dataset made available through Kaggle, the Federal Reserve Bank of St. Louis for public access to FRED economic data, and the Consumer Financial Protection Bureau for maintaining the public complaint database.
