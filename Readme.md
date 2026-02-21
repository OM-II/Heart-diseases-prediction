Heart Disease Prediction – Machine Learning Project

Overview
This project focuses on predicting the probability of heart disease using a machine learning model.
The model outputs probabilities and is evaluated using ROC-AUC, which measures the model’s ability to distinguish between patients with and without heart disease.

The solution follows an industry-level end-to-end machine learning workflow, from data preprocessing and model training to API-based deployment using FastAPI.

Problem Statement
Given patient health data, predict the likelihood of heart disease.

Target Variable: Heart Disease (0 or 1)
Output: Probability of heart disease
Evaluation Metric: Area Under the ROC Curve (ROC-AUC)

Project Structure

heart-disease-ml/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── external/
│
├── src/
│ ├── config/
│ ├── data/
│ ├── features/
│ ├── models/
│ ├── training/
│ ├── evaluation/
│ └── inference/
│
├── app/
│ ├── api/
│ ├── schemas/
│ └── main.py
│
├── models/
│ ├── v1/
│ └── v2/
│
├── tests/
│
├── docker/
│
├── .env
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md

Data
Raw Data: Original datasets before preprocessing
Processed Data: Cleaned and transformed datasets ready for modeling
External Data: Third-party or supplementary data sources

Model Development
Data preprocessing using sklearn pipelines
Feature engineering for numerical and categorical variables
Binary classification model trained to predict probabilities
Model evaluation using ROC-AUC on validation data

Model Saving
Trained models are saved as serialized artifacts (.pkl) to ensure reusability and reproducibility.
Model versioning is maintained under the models directory.

API Deployment
The trained model is deployed using FastAPI.

Example Endpoint
POST /predict
Input: Patient health features in JSON format
Output: Probability of heart disease

Installation
pip install -r requirements.txt

Running the Application
uvicorn app.main:app --reload

API documentation is available at:
http://127.0.0.1:8000/docs

Future Improvements
Advanced models such as XGBoost or LightGBM
Hyperparameter tuning
Model monitoring and retraining
CI/CD pipeline integration
Cloud deployment

Data set 

Column 	                        Data type   (ML)                      Explanation
id	                            Identifier	                        Not a feature drop
Age	                            Numerical (continuous)	            Age in years
Sex	                            Categorical (binary)	            0 = Female, 1 = Male
Chest pain  type	            Categorical (nominal)	            Types 1–4, no order
BP	                            Numerical (continuous)	            Blood pressure
Cholesterol	                    Numerical (continuous)	            Serum cholesterol
FBS over 120	                Categorical (binary)	            0 = False, 1 = True
EKG results	                    Categorical (nominal)	            Discrete  heart signals
Max HR	                        Numerical   (continuous)	         Maximum heart rate
Exercise angina	                Categorical (binary)	            0 = No, 1 = Yes
ST depression	                Numerical   (continuous)             ECG measurement
Slope of ST	                    Categorical (ordinal)	            Ordered values (1 < 2 < 3)
Number of vessels fluro	        Numerical   (discrete)	            Count (0–3)
Thallium	                    Categorical (nominal)	            3, 6, 7 are labels
Heart Disease Target            categorical ( binary)	            Presence / Absence


                ┌─────────────────────┐
                │   Load CSV Dataset  │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Identify Columns   │
                │ (by meaning, not    │
                │  only data type)    │
                └─────────┬───────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌─────────────────────┐         ┌─────────────────────┐
│   Identifier Column │         │   Feature Columns   │
│        (id)         │         │                     │
└─────────┬───────────┘         └─────────┬───────────┘
          │                                 │
          ▼                                 ▼
┌─────────────────────┐     ┌────────────────────────────┐
│   Drop Column       │     │  Separate Feature Types     │
│   (Not informative)│     │                            │
└─────────────────────┘     │  • Numerical               │
                            │  • Categorical             │
                            └─────────┬──────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ Numerical Features  │   │ Categorical Features│   │ Target Variable     │
│ (Age, BP, etc.)     │   │ (Chest pain, etc.)  │   │ (Heart Disease)     │
└─────────┬───────────┘   └─────────┬───────────┘   └─────────┬───────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌────────────────────────────┐   ┌─────────────────────┐
│ Scaling / Normalize │   │ Encoding                    │   │ Label Encoding      │
│ (StandardScaler)   │   │ • Binary → keep 0/1         │   │ Presence → 1        │
│                     │   │ • Nominal → One-Hot        │   │ Absence  → 0        │
└─────────┬───────────┘   │ • Ordinal → Ordinal Encode │   └─────────┬───────────┘
          │               └─────────┬──────────────────┘             │
          └───────────────┬─────────┴─────────┬───────────────────────┘
                          ▼                   ▼
                 ┌─────────────────────────────────┐
                 │   Combine All Processed Data     │
                 │        (Feature Matrix X)        │
                 └───────────────┬─────────────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────┐
                 │     Train ML Model               │
                 │ (Logistic / RF / SVM etc.)       │
                 └─────────────────────────────────┘

Numerical
Continuous:      Age, BP, Cholesterol, Max HR, ST depression
Discrete counts: Number of vessels fluro

Categorical
Binary:  Sex, FBS over 120, Exercise angina
Nominal: Chest pain type, EKG results, Thallium
Ordinal: Slope of ST

Target variable
Heart Disease → Binary classification

Biological ranges & risk interpretation
Age (years)
Range	    Risk meaning
< 40	    Low risk
40–55	    Moderate risk
> 55	    High risk
Risk increases with arterial aging.

Sex
Value	    Risk meaning
Female	    Lower risk (pre-menopause)
Male	    Higher risk
Hormonal protection in females reduces early risk.

Chest Pain Type
Type	    Meaning	             Risk
1	        Typical angina	     High
2	        Atypical angina	     Medium
3	        Non-anginal pain	 Low
4	        Asymptomatic	     High (dangerous because silent)

BP –        Blood Pressure (mm Hg)
Range	    Risk
< 120	    Low
120–139	    Moderate
≥ 140	    High
High BP damages artery walls.


Cholesterol (mg/dL)
Range	        Risk
< 200	        Low
200–239	        Moderate
≥ 240	        High
Higher cholesterol → plaque buildup.


FBS over 120 (Fasting Blood Sugar)
Value	                Risk
0 (≤120)	            Low
1 (>120)	            High
Indicates diabetes      risk.

EKG Results
Value	    Meaning                         Risk
0	        Normal	                        Low
1	        ST-T abnormality	            Moderate
2	        Left ventricular hypertrophy	High

Max Heart Rate (bpm)
Range	    Risk
> 160	    Low
120–160	    Moderate
< 120	    High
Low max HR may show poor heart performance.

Exercise Angina
Value	    Risk
0 (No)	    Low
1 (Yes)	    High
Chest pain during exercise is a strong warning sign.

ST Depression
Value	        Risk
0 – 1.0	        Low
1.0 – 2.0	    Moderate
> 2.0	        High
Measures heart muscle oxygen shortage.

Slope of ST
Value	    Meaning 	Risk
1	        Upsloping	Low
2	        Flat	    Moderate
3	        Downsloping	High

Number of Vessels (Fluro)
Count	    Risk
0	        Low
1–2	        Moderate
≥ 3	        High
More blocked vessels = higher disease severity.

Thallium Test
Value	    Meaning	Risk
3	        Normal	Low
6	        Fixed defect	Moderate
7	        Reversible defect	High
Shows reduced blood flow to heart muscle.

Heart Disease (Target)
Value	    Meaning
0	        Absence (Low risk)
1	        Presence (High risk)

preprocessing 
1️⃣ Numerical columns
Missing values → filled with median
Values → scaled (important for ML)

2️⃣ Categorical columns
Missing values → filled with most frequent
Converted to numbers using One-Hot Encoding

3️⃣ ColumnTransformer
Applies correct preprocessing to correct columns
Prevents data leakage