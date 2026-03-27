# Churn Prediction MLOps

\# 🔄 Customer Churn Prediction — End-to-End MLOps System



!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)

!\[FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)

!\[Docker](https://img.shields.io/badge/Docker-Containerized-blue)

!\[MLflow](https://img.shields.io/badge/MLflow-Tracking-red)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)



\## 📌 Project Overview

An end-to-end machine learning system that predicts customer churn for a telecom company. Built with production-grade tools used at top tech companies — featuring experiment tracking, REST API serving, Docker containerization, data drift monitoring and an interactive dashboard.



\*\*Dataset:\*\* Telco Customer Churn (7,043 customers, 21 features)

\*\*Best Model:\*\* XGBoost (tuned with Optuna) — AUC-ROC: 0.84+



\---



\## 🏗️ System Architecture

```

Raw Data → EDA → Feature Engineering → Model Training

&#x20;   → MLflow Tracking → FastAPI → Docker → Streamlit Dashboard

&#x20;                                       ↓

&#x20;                             Drift Monitoring (KS Test + PSI)

```



\---



\## 🚀 Key Features



\- \*\*Exploratory Data Analysis\*\* — class imbalance detection, hidden null discovery, correlation analysis

\- \*\*Feature Engineering\*\* — 6 new features created, `ChargePerService` became #1 predictor

\- \*\*Experiment Tracking\*\* — MLflow logs all runs with params, metrics and artifacts

\- \*\*Hyperparameter Tuning\*\* — Optuna runs 50 trials to find optimal XGBoost params

\- \*\*REST API\*\* — FastAPI with `/predict` and `/predict/batch` endpoints

\- \*\*Containerization\*\* — Fully Dockerized for deployment anywhere

\- \*\*Drift Monitoring\*\* — KS Test + PSI scores detect feature distribution shifts

\- \*\*Interactive Dashboard\*\* — Streamlit app with prediction, drift monitoring and model info pages



\---



\## 📊 Model Performance



| Model | AUC-ROC | Recall | F1 |

|-------|---------|--------|----|

| Logistic Regression (baseline) | 0.8477 | 0.7834 | 0.6162 |

| Random Forest | 0.8230 | 0.4733 | 0.5339 |

| XGBoost (baseline) | 0.8411 | 0.7727 | 0.6262 |

| \*\*XGBoost (Optuna tuned)\*\* | \*\*0.84+\*\* | \*\*0.77+\*\* | \*\*0.62+\*\* |



\---



\## 🔑 Key Findings from EDA



\- \*\*Class Imbalance\*\* — 73.5% No Churn vs 26.5% Yes Churn

\- \*\*Contract Type\*\* — Month-to-month customers churn 3x more than two-year customers

\- \*\*Fiber Optic\*\* — Fiber optic users churn significantly more than DSL users

\- \*\*Tenure\*\* — Long-term customers (>36 months) have only \~11% churn rate



\---



\## 🛠️ Tech Stack



| Category | Tools |

|----------|-------|

| ML \& Data | Python, Pandas, Scikit-learn, XGBoost |

| Experiment Tracking | MLflow |

| Hyperparameter Tuning | Optuna |

| API Serving | FastAPI, Uvicorn, Pydantic |

| Containerization | Docker |

| Drift Monitoring | SciPy (KS Test), PSI |

| Dashboard | Streamlit |

| Version Control | Git, GitHub |



\---



\## 📁 Project Structure

```

churn-prediction-mlops/

├── data/                          # Raw dataset

├── notebooks/

│   ├── 01\_EDA.ipynb               # Exploratory data analysis

│   ├── 02\_Preprocessing\_Baseline.ipynb  # Baseline models

│   ├── 03\_Feature\_Engineering.ipynb     # Feature engineering

│   ├── 04\_MLflow\_Tracking.ipynb         # Experiment tracking

│   ├── 05\_Optuna\_Tuning.ipynb           # Hyperparameter tuning

│   └── 08\_Drift\_Monitoring.ipynb        # Drift detection

├── src/

│   ├── app.py                     # FastAPI application

│   └── dashboard.py               # Streamlit dashboard

├── models/                        # Saved model artifacts

├── reports/                       # Drift reports \& visualizations

├── Dockerfile                     # Docker configuration

├── requirements.txt               # Python dependencies

└── README.md                      # Project documentation

```



\---



\## ⚡ Quick Start



\### 1. Clone the repository

```bash

git clone https://github.com/smukherjee1116-lgtm/Churn-prediction-mlops.git

cd Churn-prediction-mlops

```



\### 2. Install dependencies

```bash

pip install -r requirements.txt

```



\### 3. Run the FastAPI

```bash

cd src

uvicorn app:app --reload

```

Visit \*\*http://127.0.0.1:8000/docs\*\* for the Swagger UI



\### 4. Run with Docker

```bash

docker build -t churn-prediction-api .

docker run -p 8000:8000 churn-prediction-api

```



\### 5. Run the Streamlit Dashboard

```bash

cd src

streamlit run dashboard.py

```

Visit \*\*http://localhost:8501\*\*



\---



\## 🔮 API Usage



\### Single Prediction

```bash

curl -X POST "http://127.0.0.1:8000/predict" \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "tenure": 2,

&#x20;   "MonthlyCharges": 70.5,

&#x20;   "TotalCharges": 141.0,

&#x20;   "SeniorCitizen": 0,

&#x20;   "gender": 1,

&#x20;   "Partner": 0,

&#x20;   "Dependents": 0,

&#x20;   "PhoneService": 1,

&#x20;   "PaperlessBilling": 1,

&#x20;   "NumServices": 1,

&#x20;   "IsMonthToMonth": 1,

&#x20;   "IsHighValue": 1

&#x20; }'

```



\### Sample Response

```json

{

&#x20; "churn\_prediction": 1,

&#x20; "churn\_probability": 0.7823,

&#x20; "risk\_level": "High",

&#x20; "message": "Customer likely to churn!"

}

```



\---



\## 📈 Drift Monitoring



The system monitors 6 key features using:

\- \*\*KS Test\*\* — detects distribution shifts statistically

\- \*\*PSI Score\*\* — measures population stability (PSI > 0.2 = retrain!)



| Feature | PSI Score | Status |

|---------|-----------|--------|

| tenure | 2.73 | 🔴 Drift |

| MonthlyCharges | 0.31 | 🔴 Drift |

| NumServices | 1.11 | 🔴 Drift |

| TotalCharges | 0.009 | 🟢 Stable |

| AvgMonthlySpend | 0.0 | 🟢 Stable |

| ChargePerService | 0.006 | 🟢 Stable |



\---



\## 👤 Author

\*\*Soham Mukherjee\*\*

\- GitHub: \[@smukherjee1116-lgtm](https://github.com/smukherjee1116-lgtm)



\---



\*Built as part of a portfolio project targeting Data Scientist roles at top tech companies.\*

