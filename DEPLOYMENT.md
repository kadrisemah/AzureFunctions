# 🚀 Deployment Guide - Azure Functions ML Pipelines

**Deployment guide for production environments**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Azure Deployment](#azure-deployment)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Rollback Procedures](#rollback-procedures)

---

## 🔧 Prerequisites

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.11 or 3.10 | Runtime environment |
| **Azure Functions Core Tools** | 4.x | Local development & deployment |
| **Azure CLI** | Latest | Azure resource management |
| **Azurite** | Latest | Local storage emulation |
| **ODBC Driver** | 17 or 18 for SQL Server | Database connectivity |
| **Git** | Latest | Version control |

### Installation Commands

```bash
# Azure CLI
# Windows: Download from https://aka.ms/installazurecliwindows
# Linux/macOS:
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Azurite (Storage Emulator)
npm install -g azurite

# ODBC Driver for SQL Server
# Download: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
```

### Azure Resources Required

- **Azure Function App** (Consumption or Premium plan)
- **Azure SQL Database** (Source & Target)
- **Application Insights** (Recommended for monitoring)
- **Azure Storage Account** (Automatically created with Function App)

---

## 💻 Local Development Setup

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd azure_functions_clean
```

### 2. Choose Your Pipeline

#### Option A: Top-Up Pipeline

```bash
cd topup
```

#### Option B: SQL/SAL Pipeline

```bash
cd sqlsal
```

#### Option C: Client Conversion Pipeline

```bash
cd client_conversion
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows CMD:
.\.venv\Scripts\activate.bat

# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Local Settings

```bash
# Copy template
cp local.settings.json.template local.settings.json

# Edit local.settings.json with your credentials
# Use your preferred editor (VS Code, Notepad++, vim, etc.)
code local.settings.json  # VS Code
notepad local.settings.json  # Windows Notepad
```

**Example local.settings.json:**

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "SOURCE_SERVER": "your-server.database.windows.net",
    "SOURCE_DATABASE": "your-source-db",
    "SOURCE_USER": "your-username",
    "SOURCE_PASSWORD": "your-password",
    "TARGET_SERVER": "your-server.database.windows.net",
    "TARGET_DATABASE": "your-target-db",
    "TARGET_USER": "your-username",
    "TARGET_PASSWORD": "your-password"
  }
}
```

⚠️ **CRITICAL:** Never commit `local.settings.json` to version control!

### 5. Start Local Development

**Terminal 1 - Start Azurite:**
```bash
azurite
```

**Terminal 2 - Start Functions:**
```bash
func start
```

The functions will be available at: `http://localhost:7071`

---

## ☁️ Azure Deployment

### Method 1: Direct Deployment (Recommended)

#### Step 1: Login to Azure

```bash
az login
```

#### Step 2: Create Function App (First Time Only)

```bash
# Set variables
RESOURCE_GROUP="rg-ml-pipelines"
LOCATION="eastus"
STORAGE_ACCOUNT="stmlpipelines$(date +%s)"  # Must be unique
FUNCTION_APP_NAME="func-topup-pipeline"  # Must be globally unique

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create storage account
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Create Function App (Consumption Plan - Python 3.11)
az functionapp create \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME \
  --storage-account $STORAGE_ACCOUNT \
  --consumption-plan-location $LOCATION \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux
```

#### Step 3: Deploy Code

```bash
# Navigate to pipeline folder
cd topup  # or cd sqlsal

# Deploy
func azure functionapp publish $FUNCTION_APP_NAME --python
```

#### Step 4: Configure Application Settings

```bash
# Set database credentials (one-time setup)
az functionapp config appsettings set \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    SOURCE_SERVER="your-server.database.windows.net" \
    SOURCE_DATABASE="your-database" \
    SOURCE_USER="your-username" \
    SOURCE_PASSWORD="your-password" \
    TARGET_SERVER="your-target-server.database.windows.net" \
    TARGET_DATABASE="your-target-database" \
    TARGET_USER="your-target-username" \
    TARGET_PASSWORD="your-target-password"
```

### Method 2: CI/CD with GitHub Actions (Advanced)

Create `.github/workflows/deploy-topup.yml`:

```yaml
name: Deploy Top-Up Pipeline

on:
  push:
    branches:
      - main
    paths:
      - 'topup/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          cd topup
          pip install -r requirements.txt

      - name: Deploy to Azure
        uses: Azure/functions-action@v1
        with:
          app-name: ${{ secrets.AZURE_FUNCTIONAPP_NAME }}
          package: ./topup
          publish-profile: ${{ secrets.AZURE_FUNCTIONAPP_PUBLISH_PROFILE }}
```

Similarly, create `.github/workflows/deploy-sqlsal.yml` and `.github/workflows/deploy-client-conversion.yml` for the other pipelines, adjusting the paths:
- SQLSAL: `paths: - 'sqlsal/**'` and `package: ./sqlsal`
- Client Conversion: `paths: - 'client_conversion/**'` and `package: ./client_conversion`

---

## ⚙️ Configuration

### Environment Variables

#### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SOURCE_SERVER` | Source database server | `prod-db.database.windows.net` |
| `SOURCE_DATABASE` | Source database name | `hubspot_data` |
| `SOURCE_USER` | Source database username | `readonly_user` |
| `SOURCE_PASSWORD` | Source database password | `<secure-password>` |
| `TARGET_SERVER` | Target database server | `analytics-db.database.windows.net` |
| `TARGET_DATABASE` | Target database name | `ml_predictions` |
| `TARGET_USER` | Target database username | `ml_writer` |
| `TARGET_PASSWORD` | Target database password | `<secure-password>` |

#### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FUNCTIONS_WORKER_RUNTIME` | Runtime | `python` |
| `AzureWebJobsStorage` | Storage connection | Auto-configured |
| `APPINSIGHTS_INSTRUMENTATIONKEY` | Monitoring | Auto-configured |

### Schedule Configuration

Schedules are defined in each function's `function.json`:

```json
{
  "schedule": "0 0 6 1 * *"  // Cron expression
}
```

**Cron Format:** `{second} {minute} {hour} {day} {month} {day-of-week}`

**Examples:**
- `0 0 6 1 * *` - 6:00 AM on the 1st of every month
- `0 30 6 1 * *` - 6:30 AM on the 1st of every month
- `0 0 8 * * *` - 8:00 AM every day

---

## 🧪 Testing

### Local Testing

#### Manual Trigger (PowerShell)

```powershell
# Top-Up Data Preparation
Invoke-RestMethod -Uri "http://localhost:7071/admin/functions/data_preparation" `
  -Method POST `
  -ContentType "application/json" `
  -Body "{}"

# Top-Up Model Training
Invoke-RestMethod -Uri "http://localhost:7071/admin/functions/model_training" `
  -Method POST `
  -ContentType "application/json" `
  -Body "{}"

# SQL/SAL Data Preparation
Invoke-RestMethod -Uri "http://localhost:7071/admin/functions/sqlsal_data_preparation" `
  -Method POST `
  -ContentType "application/json" `
  -Body "{}"

# SQL/SAL Model Training
Invoke-RestMethod -Uri "http://localhost:7071/admin/functions/sqlsal_model_training" `
  -Method POST `
  -ContentType "application/json" `
  -Body "{}"
```

#### Manual Trigger (Bash/curl)

```bash
# Data Preparation
curl -X POST http://localhost:7071/admin/functions/data_preparation \
  -H "Content-Type: application/json" \
  -d "{}"
```

### Azure Testing

```bash
# Trigger via Azure CLI
az functionapp function invoke \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME \
  --function-name data_preparation
```

### Validation Checklist

After deployment, verify:

- [ ] Functions appear in Azure Portal
- [ ] Application Settings are configured
- [ ] Functions execute without errors
- [ ] Data is written to target database
- [ ] Logs appear in Application Insights
- [ ] Timer triggers are scheduled correctly

---

## 📊 Monitoring

### Application Insights

**View Logs:**
1. Azure Portal → Function App → Application Insights
2. Navigate to **Transaction search** or **Logs**

**Query Examples:**

```kusto
// All function executions in last 24 hours
traces
| where timestamp > ago(24h)
| where message contains "Function"
| project timestamp, severityLevel, message

// Errors only
exceptions
| where timestamp > ago(24h)
| project timestamp, type, outerMessage

// Performance metrics
requests
| where timestamp > ago(24h)
| summarize avg(duration), max(duration) by name
```

### Alerts (Recommended)

Create alerts for:
- Function failures
- Execution duration > threshold
- Database connection failures

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Module not found" Error

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

#### 2. Database Connection Failed

**Check:**
- Firewall rules allow Azure services
- Credentials are correct
- Network connectivity

**Test Connection:**
```python
import pyodbc
conn_str = "Driver={ODBC Driver 18 for SQL Server};Server=your-server.database.windows.net;Database=your-db;Uid=your-user;Pwd=your-pwd"
conn = pyodbc.connect(conn_str)
```

#### 3. Function Timeout

**Solution:**
- Increase timeout in `host.json`: `"functionTimeout": "01:00:00"`
- Consider Premium plan for longer executions

#### 4. Optuna Trial Failures (SQL/SAL)

**Check:**
- Data quality and consistency
- Sufficient training samples (min 100+)
- Feature columns exist

---

## ⏮️ Rollback Procedures

### Rollback to Previous Version

```bash
# List deployment history
az functionapp deployment list \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME

# Rollback to specific deployment
az functionapp deployment source config-zip \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME \
  --src <previous-deployment.zip>
```

### Emergency Stop

```bash
# Stop Function App
az functionapp stop \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME

# Disable specific function
az functionapp function disable \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME \
  --function-name model_training
```

---

## 📞 Support & Resources

- **Azure Functions Documentation:** https://docs.microsoft.com/azure/azure-functions/
- **Python Developer Guide:** https://docs.microsoft.com/azure/azure-functions/functions-reference-python
- **CatBoost Documentation:** https://catboost.ai/docs/
- **Optuna Documentation:** https://optuna.readthedocs.io/

---

## ✅ Deployment Checklist

Before deploying to production:

- [ ] All tests passing locally
- [ ] Database credentials configured securely
- [ ] `.gitignore` properly configured
- [ ] `local.settings.json` NOT in repository
- [ ] Application Insights enabled
- [ ] Alerts configured
- [ ] Backup/rollback plan documented
- [ ] Stakeholders notified of deployment schedule
- [ ] Monitoring dashboard ready

---
