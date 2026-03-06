
---

# Log-Warden: Self-Healing DevOps Agent

Log-Warden is a DevOps automation system that analyzes system logs, predicts incident severity, estimates time-to-failure, and generates remediation scripts using a Large Language Model (LLM).

The system integrates **Machine Learning**, **LLM-based remediation**, **REST APIs**, and **Docker containerization** to demonstrate a practical automated incident-response pipeline.

---

# Key Features

* Automated log analysis
* Severity classification using Machine Learning
* Time-to-failure prediction
* LLM-generated remediation scripts
* REST API built with Flask
* Containerized deployment using Docker
* Unit tests with pytest

---

# Architecture Overview

```text
Client
  │
  │ POST /analyze
  ▼
Flask API (app/main.py)
  │
  ├── ML Pipeline
  │      ├ preprocessing
  │      ├ vectorization
  │      ├ severity classification
  │      └ time-to-failure regression
  │
  └── LLM Agent (Groq)
         └ remediation script generation
```

---

# Project Structure

```text
log-warden/
│
├── app/
│   ├── agent/
│   │   └── groq_agent.py
│   │
│   ├── ml/
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   └── preprocessing.py
│   │
│   ├── main.py
│   └── schemas.py
│
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   └── vectorizer.pkl
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_agent.py
│   └── test_ml.py
│
├── generate_dataset.py
├── train.py
├── logs.csv
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

---

# Machine Learning Pipeline

The ML pipeline performs two main tasks:

### Severity Classification

Predicts whether a log indicates a critical incident.

Model used:

```text
LogisticRegression
```

Output:

```text
high_severity = 0 or 1
```

---

### Time-to-Failure Prediction

Estimates how soon a system failure may occur.

Model used:

```text
RandomForestRegressor
```

Output:

```text
time_to_failure_minutes
```

---

### Log Preprocessing

Logs are cleaned using:

* timestamp removal
* IP address removal
* punctuation filtering
* stopword removal
* lemmatization using spaCy

---

# LLM Remediation Agent

If a log is classified as **high severity**, the system triggers an LLM agent.

LLM used:

```text
Groq llama-3.1-8b-instant
```

The agent generates a **safe Python remediation script**.

Safety constraints include:

* system state validation
* logging usage
* no destructive operations

Forbidden commands include:

```text
kill -9
rm -rf
fsck -f
systemctl restart
```

---

# API Endpoints

## Health Check

```
GET /health
```

Response

```json
{
  "status": "running"
}
```

---

## Analyze Log

```
POST /analyze
```

Request body

```json
{
  "log_text": "ERROR database connection lost"
}
```

Example response

```json
{
  "status": "success",
  "analysis": {
    "log_text": "ERROR database connection lost",
    "predictions": {
      "high_severity": true,
      "time_to_failure_minutes": 21.09
    }
  },
  "agent": {
    "decision": "LLM triggered because high_severity == 1",
    "model": "llama-3.1-8b-instant"
  },
  "remediation": {
    "generated": true,
    "issue_category": "Database Connectivity Issue",
    "confidence": 0.85,
    "script_language": "python",
    "script_lines": [...]
  }
}
```

---

# Local Setup

## Clone repository

```bash
git clone https://github.com/<your-username>/log-warden.git
cd log-warden
```

---

## Create virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

Install spaCy model

```bash
python -m spacy download en_core_web_sm
```

---

## Configure environment variables

Create `.env`

```text
GROQ_API_KEY=your_api_key_here
```

---

## Train models

```bash
python train.py
```

---

## Run the API locally

```bash
python -m flask --app app.main run
```

Server runs at:

```
http://localhost:5000
```

---

# Running with Docker

## Build Docker image

```bash
docker build -t log-warden .
```

---

## Run container

```bash
docker run --env-file .env -p 5000:5000 log-warden
```

---

# Testing the API

## Health Endpoint

### Windows PowerShell

```powershell
Invoke-RestMethod http://localhost:5000/health | ConvertTo-Json
```

### Linux / Mac

```bash
curl http://localhost:5000/health
```

---

## Analyze Endpoint

### Windows PowerShell

```powershell
Invoke-RestMethod -Method POST http://localhost:5000/analyze -ContentType "application/json" -Body '{"log_text":"ERROR database connection lost"}' | ConvertTo-Json -Depth 10
```

### Linux / Mac

```bash
curl -X POST http://localhost:5000/analyze \
-H "Content-Type: application/json" \
-d '{"log_text":"ERROR database connection lost"}'
```

---

# Running Tests

```bash
pytest
```

Tests cover:

* API endpoints
* ML pipeline
* LLM remediation logic

---

# Dataset Generation

Synthetic logs can be generated with:

```bash
python generate_dataset.py
```

This produces:

```
logs.csv
```

Used for model training.

---

# Technologies Used

| Component        | Technology   |
| ---------------- | ------------ |
| API              | Flask        |
| Machine Learning | scikit-learn |
| Text Processing  | spaCy        |
| LLM              | Groq         |
| Containerization | Docker       |
| Testing          | pytest       |

---

# Future Improvements

Possible enhancements include:

* real-time log ingestion
* Kubernetes self-healing
* monitoring dashboards
* streaming log pipelines
* reinforcement learning for remediation ranking

---

# Author

Vishal Telukoti

---

