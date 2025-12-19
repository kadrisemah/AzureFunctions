# ⚡ Quick Start Guide

---

## 🎯 Choose Your Pipeline

This project contains **two independent ML pipelines**:

### 1. **Top-Up Prediction** (`topup/`)
- Predicts client top-up probability
- Monthly execution (1st of month, 6:00 AM & 6:30 AM)
- ~30 minutes total runtime

### 2. **SQL/SAL Lead Scoring** (`sqlsal/`)
- Scores SQL/SAL prospects with ML + NLP
- Monthly execution (1st of month, 8:00 AM & 8:30 AM)
- ~45-60 minutes total runtime

---

## 🚀 Local Development (Step-by-Step)

### Prerequisites Check

```bash
# Verify installations
python --version        # Should be 3.11 or 3.10
func --version         # Should be 4.x
azurite --version      # Should be installed
az --version           # Should be installed
```

Missing something? See [DEPLOYMENT.md](DEPLOYMENT.md#prerequisites) for installation instructions.

---

### Option A: Top-Up Pipeline

```bash
# 1. Navigate to topup folder
cd topup

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp local.settings.json.template local.settings.json
# Edit local.settings.json with your database credentials

# 4. Start Azurite (Terminal 1)
azurite

# 5. Start Functions (Terminal 2)
func start

# 6. Trigger functions (Terminal 3 - PowerShell)
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/data_preparation -Method POST -ContentType "application/json" -Body "{}"
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/model_training -Method POST -ContentType "application/json" -Body "{}"
```

---

### Option B: SQL/SAL Pipeline

```bash
# 1. Navigate to sqlsal folder
cd sqlsal

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp local.settings.json.template local.settings.json
# Edit local.settings.json with your database credentials

# 4. Start Azurite (Terminal 1)
azurite

# 5. Start Functions (Terminal 2)
func start

# 6. Trigger functions (Terminal 3 - PowerShell)
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/sqlsal_data_preparation -Method POST -ContentType "application/json" -Body "{}"
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/sqlsal_model_training -Method POST -ContentType "application/json" -Body "{}"
```

---

## ☁️ Azure Deployment (Quick)

### First-Time Setup

```bash
# 1. Login to Azure
az login

# 2. Set variables (customize these)
RESOURCE_GROUP="rg-ml-pipelines"
LOCATION="eastus"
FUNCTION_APP_NAME="func-ml-pipeline-$(date +%s)"  # Unique name

# 3. Create resources
az group create --name $RESOURCE_GROUP --location $LOCATION

# 4. Create Function App
az functionapp create \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME \
  --storage-account mlpipelinestorage \
  --consumption-plan-location $LOCATION \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux
```

### Deploy Top-Up Pipeline

```bash
cd topup
func azure functionapp publish $FUNCTION_APP_NAME --python
```

### Deploy SQL/SAL Pipeline

```bash
cd sqlsal
func azure functionapp publish $FUNCTION_APP_NAME --python
```

### Configure Credentials

```bash
az functionapp config appsettings set \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    SOURCE_SERVER="your-server.database.windows.net" \
    SOURCE_DATABASE="your-database" \
    SOURCE_USER="your-username" \
    SOURCE_PASSWORD="your-password" \
    TARGET_SERVER="your-server.database.windows.net" \
    TARGET_DATABASE="your-database" \
    TARGET_USER="your-username" \
    TARGET_PASSWORD="your-password"
```

---

## 📊 Expected Outputs

### Top-Up Pipeline Creates:
- `dbo.topup_training_dataset` (~50,000 rows)
- `dbo.topup_model_predictions` (~800 rows)
- `dbo.topup_rm_action_list` (~200 rows)
- `dbo.topup_feature_importance` (~20 rows)

### SQL/SAL Pipeline Creates:
- `dbo.sqlsal_training_dataset` (~10,000-20,000 rows)
- `dbo.sqlsal_model_predictions` (~5,000-10,000 rows)
- `dbo.sqlsal_rm_action_list` (~500-2,000 rows)
- `dbo.sqlsal_feature_importance` (~50-100 rows)
- `dbo.sqlsal_model_metadata` (1 row per run)

---

## ⏱️ Execution Times

| Pipeline | Data Prep | Model Training | Total |
|----------|-----------|----------------|-------|
| **Top-Up** | ~10 min | ~20 min | ~30 min |
| **SQL/SAL** | ~15-20 min | ~30-40 min | ~45-60 min |

---

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "Cannot connect to database"
- Check credentials in `local.settings.json`
- Verify firewall allows your IP
- Test connection with Azure Data Studio / SSMS

### Issue: "Port 7071 already in use"
```bash
# Stop other func instances
func stop
# Or use different port
func start --port 7072
```

### Issue: "Azurite connection refused"
```bash
# Start Azurite first
azurite
```

---

## 📚 Next Steps

- **Full Documentation:** See [README.md](README.md)
- **Deployment Guide:** See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Pipeline Details:**
  - Top-Up: [topup/README.md](topup/README.md)
  - SQL/SAL: [sqlsal/README.md](sqlsal/README.md)

---

## 🔒 Security Reminder

⚠️ **NEVER** commit `local.settings.json` to version control!

It contains sensitive credentials. Always use:
- `local.settings.json.template` for version control
- Azure Key Vault for production secrets

---
