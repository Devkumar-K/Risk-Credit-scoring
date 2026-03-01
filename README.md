# 💳 Unified Credit Risk System

## Overview
A Credit Risk Analysis project utilizing a Support Vector Machine (SVM) built entirely from scratch with NumPy. The project handles dataset correction, class imbalance, model training, hyperparameter tuning, and comprehensive evaluation (including ROC-AUC, confusion matrices, and feature importance). Finally, it provides a unified Streamlit web application for interactive, intuitive predictions.

This project covers two key financial risk prediction tasks:
1. **Credit Card Approval**
2. **Credit Default Prediction**

## Key Features
- **Custom Linear SVM Model:** Designed from scratch using optimization.
- **Two Distinct Prediction Tasks:** Handles logic, data transformations, and model evaluations for both Credit Card Approval and Credit Default Prediction.
- **Detailed Evaluation & Visualization:** Automatically generates comprehensive visualization artifacts such as confusion matrices, ROC curves, decision score distributions, and feature importance bar plots.
- **Interactive Unified UI:** A Streamlit application built to seamlessly run the pre-trained SVM models. You can select a task, input manual features, and interactively view real-time score interpretations along with evaluation metrics.

## Project Structure
- `card_approval_augmented.csv`: Dataset for the credit card approval task.
- `credefault.csv`: Dataset for the credit default prediction task.
- `cardaprov.py`: Script to train and evaluate the Credit Card Approval SVM model.
- `credef.py`: Script to train and evaluate the Credit Default SVM model.
- `streamlit_op.py`: The unified Streamlit interactive dashboard.
- `model/`: Automatically created directory that will store the trained JSON models (`card_approval_model.json`, `credit_default_model.json`).
- `figures/`: Automatically created directory that will store the plotted performance figures.

## Requirements
Ensure you have the following installed to run the codes:
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `streamlit`
- `Pillow` (PIL)

You can mostly install them via `pip`:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit Pillow
```

## Usage Instructions

### 1. Training the Models
Before firing up the interactive application, make sure to train the models in order to generate the required model artifacts (json format) and performance evaluation figures.
```bash
# 1. Train the Credit Card Approval Model
python cardaprov.py

# 2. Train the Credit Default Model
python credef.py
```
Upon completion, models and plots will be securely saved into `model/` and `figures/` directories respectively.

### 2. Launching the Web Application
Once the models are successfully trained, you can launch the interactive Streamlit application to query the models:
```bash
streamlit run streamlit_op.py
```
This action will launch a local server running the application in your browser. You can select either predicting "Card Approval" or "Credit Default", interactively tune numeric features on the sidebar, and quickly witness decision predictions with accompanying visual insights.
