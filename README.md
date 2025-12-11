# Azure Functions - Top-Up Prediction ML Pipeline

Automated ETL pipeline that predicts client top-up probability using CatBoost ML model.

## 📋 What This Does

**2 Azure Functions:**
1. **data_preparation** - Extracts HubSpot data, creates monthly panel dataset
2. **model_training** - Trains CatBoost model, generates predictions and action list

**Runs Monthly:** 1st of month at 6:00 AM (data prep) and 6:30 AM (model training)

---

## 🛠️ Setup Instructions

### 1. Prerequisites

- **Python 3.11** (or 3.10)
- **Azure Functions Core Tools 4.x**
- **Azurite** (storage emulator)
- **ODBC Driver 17 for SQL Server**

### 2. Install Tools

**Azure Functions Core Tools:**
```bash
npm install -g azure-functions-core-tools@4
```

**Azurite:**
```bash
npm install -g azurite
```

**ODBC Driver 17:**
Download from: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

### 3. Install Python Packages

```bash
cd azure_functions
pip install -r requirements.txt
```

### 4. Configure Database Credentials

**Copy template:**
```bash
cp local.settings.json.template local.settings.json
```

**Edit `local.settings.json` with your database credentials:**
```json
{
  "Values": {
    "SOURCE_SERVER": "your-server.database.windows.net",
    "SOURCE_DATABASE": "your-database",
    "SOURCE_USER": "your-username",
    "SOURCE_PASSWORD": "your-password",
    "TARGET_SERVER": "your-target-server.database.windows.net",
    "TARGET_DATABASE": "your-target-database",
    "TARGET_USER": "your-target-username",
    "TARGET_PASSWORD": "your-target-password"
  }
}
```

⚠️ **NEVER commit `local.settings.json` to git!** (already in .gitignore)

---

## 🚀 Run Locally

### Terminal 1: Start Azurite
```bash
azurite
```

### Terminal 2: Start Functions
```bash
func start
```

### Terminal 3: Trigger Functions
```bash
# Test data preparation
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/data_preparation -Method POST -ContentType "application/json" -Body "{}"

# Test model training
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/model_training -Method POST -ContentType "application/json" -Body "{}"
```

---

## 📊 Outputs

### SQL Tables Created:

1. **`dbo.topup_training_dataset`** (~50,000 rows)
   - Training data with client history and features

2. **`dbo.topup_model_predictions`** (~800 rows)
   - Predictions for all clients

3. **`dbo.topup_rm_action_list`** (~200 rows)
   - High-probability clients for relationship managers

4. **`dbo.topup_feature_importance`** (~20 rows)
   - Feature importance rankings

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found: pandas" | Run `pip install -r requirements.txt` |
| "Cannot connect to database" | Check credentials in local.settings.json |
| "ODBC driver not found" | Install ODBC Driver 17 for SQL Server |
| "Port 7071 in use" | Stop other func instances or use different port |
| "Storage connection refused" | Start Azurite first |

---

## 📁 Project Structure

```
azure_functions/
├── data_preparation/          # Function 1: Data ETL
│   ├── __init__.py           # Main data preparation logic
│   └── function.json         # Timer trigger config
├── model_training/            # Function 2: ML Training
│   ├── __init__.py           # Model training logic
│   └── function.json         # Timer trigger config
├── host.json                  # Function App configuration
├── requirements.txt           # Python dependencies
├── local.settings.json        # Local credentials (gitignored)
└── local.settings.json.template  # Template for credentials
```

---

## ⏰ Automation Schedule

**Production (Azure):**
- **Data Preparation:** 1st of month, 6:00 AM UTC (`0 0 6 1 * *`)
- **Model Training:** 1st of month, 6:30 AM UTC (`0 30 6 1 * *`)

---

## 🚀 Deploy to Azure

1. **Login:**
   ```bash
   az login
   ```

2. **Deploy:**
   ```bash
   func azure functionapp publish YOUR_FUNCTION_APP_NAME --python
   ```

3. **Configure credentials in Azure Portal:**
   - Function App → Configuration → Application settings
   - Add 8 database credential settings

---

## 📝 Notes

- **Local testing uses real database** (same as production)
- **Health monitor disabled** for long-running ML training
- **Function timeout:** 1 hour (for model training)
- **Expected duration:** Data prep ~10 min, Model training ~20 min

---

## 🔒 Security

- `local.settings.json` is gitignored (contains passwords)
- Never commit real credentials
- Use Azure Key Vault for production (optional improvement)

---

## 📞 Support

If you have issues, check:
1. All tools installed correctly
2. Python version is 3.11 or 3.10
3. Database credentials are correct
4. Azurite is running before func start

---

**Good luck testing!** 🚀
