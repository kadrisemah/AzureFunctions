# Azure Functions - ML Pipelines

This project contains two separate ML pipelines for lead scoring and prediction, organized into independent subfolders.

## 📁 Project Structure

```
azure_functions_clean/
├── topup/                          # Top-Up Prediction Pipeline
│   ├── data_preparation/           # ETL for top-up data
│   ├── model_training/             # CatBoost model training
│   ├── host.json
│   ├── requirements.txt
│   ├── local.settings.json.template
│   └── README.md
│
├── sqlsal/                         # SQL/SAL Lead Scoring Pipeline
│   ├── sqlsal_data_preparation/    # ETL with NLP processing
│   ├── sqlsal_model_training/      # CatBoost + Optuna optimization
│   ├── host.json
│   ├── requirements.txt
│   ├── local.settings.json.template
│   ├── SQLSAL 1.ipynb             # Original notebook (reference)
│   └── README.md
│
├── .git/                           # Git repository
├── .gitignore                      # Main gitignore
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Top-Up Pipeline

```bash
cd topup
pip install -r requirements.txt
cp local.settings.json.template local.settings.json
# Edit local.settings.json with credentials
func start
```

### SQL/SAL Pipeline

```bash
cd sqlsal
pip install -r requirements.txt
cp local.settings.json.template local.settings.json
# Edit local.settings.json with credentials
func start
```

---

## 📋 Pipeline Comparison

| Feature | Top-Up | SQL/SAL |
|---------|--------|---------|
| **Purpose** | Predict client top-up probability | Score SQL/SAL prospects |
| **Data Sources** | HubSpot deals & contacts | HubSpot contacts, meetings, calls |
| **ML Model** | CatBoost Classifier | CatBoost + Optuna optimization |
| **Special Features** | Monthly panel data | NLP clustering, 60+ features |
| **Schedule** | 6:00 AM & 6:30 AM | 8:00 AM & 8:30 AM |
| **Duration** | ~30 min total | ~45-60 min total |
| **Output Rows** | ~50,000 training | ~10,000-20,000 training |

---

## 🛠️ Prerequisites

- **Python 3.11** (or 3.10)
- **Azure Functions Core Tools 4.x**
- **Azurite** (storage emulator)
- **ODBC Driver 17 or 18 for SQL Server**

### Install Tools

```bash
# Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Azurite
npm install -g azurite
```

---

## ⏰ Automation Schedule (Production)

### Top-Up Pipeline
- **Data Preparation:** 1st of month, 6:00 AM UTC (`0 0 6 1 * *`)
- **Model Training:** 1st of month, 6:30 AM UTC (`0 30 6 1 * *`)

### SQL/SAL Pipeline
- **Data Preparation:** 1st of month, 8:00 AM UTC (`0 0 8 1 * *`)
- **Model Training:** 1st of month, 8:30 AM UTC (`0 30 8 1 * *`)

---

## 🚀 Deploy to Azure

Each pipeline can be deployed independently:

```bash
# Deploy Top-Up
cd topup
func azure functionapp publish YOUR_TOPUP_FUNCTION_APP_NAME --python

# Deploy SQL/SAL
cd sqlsal
func azure functionapp publish YOUR_SQLSAL_FUNCTION_APP_NAME --python
```

After deployment, configure database credentials in Azure Portal:
- Function App → Configuration → Application settings
- Add: `SOURCE_SERVER`, `SOURCE_DATABASE`, `SOURCE_USER`, `SOURCE_PASSWORD`
- Add: `TARGET_SERVER`, `TARGET_DATABASE`, `TARGET_USER`, `TARGET_PASSWORD`

---

## 📊 Output Tables

### Top-Up Pipeline
- `dbo.topup_training_dataset`
- `dbo.topup_model_predictions`
- `dbo.topup_rm_action_list`
- `dbo.topup_feature_importance`

### SQL/SAL Pipeline
- `dbo.sqlsal_training_dataset`
- `dbo.sqlsal_model_predictions`
- `dbo.sqlsal_rm_action_list`
- `dbo.sqlsal_feature_importance`
- `dbo.sqlsal_model_metadata`

---

## 📝 Notes

- Each pipeline is **completely independent** with its own configuration
- Both pipelines use the same database credentials but different tables
- Each can be tested and deployed separately
- Each has its own `requirements.txt` with pipeline-specific dependencies

---

## 🔒 Security

- **Never commit `local.settings.json`** files (gitignored)
- Use `local.settings.json.template` as reference
- Model files (*.cbm) are gitignored
- Consider Azure Key Vault for production credentials

---

## 📚 Documentation

- See `topup/README.md` for Top-Up pipeline details
- See `sqlsal/README.md` for SQL/SAL pipeline details
- See `sqlsal/SQLSAL 1.ipynb` for original notebook reference

---
