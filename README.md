Customer Churn Prediction – Complete App and Notebook
This project provides a comprehensive solution for customer churn analysis and prediction using both an interactive Streamlit dashboard and a Jupyter notebook workflow. Both tools function with structured, tabular data and are tailored for classification problems such as predicting churn in a banking context.

Project Structure
1. Streamlit Dashboard (xai_app.py)
An interactive analytics and explainability web application, allowing users to upload their dataset and explore churn insights through visualizations, metrics, and model interpretability tools.

Main Features
User-Friendly UI: Simple file upload and interactive data exploration.

Filtering: Option to analyze churn by country (Geography filter).

Exploratory Analytics:

Key customer metrics (counts, averages, churn rate).

Visualizations of important features: credit score, age, balance, salary, tenure, gender, membership, and churn status.

Side-by-side country and group comparisons.

Demographics & Geography:

Pie, histogram, and bar plots for age, gender, and regional distribution.

Financial Analysis:

Faceted plots for balance, salary, and credit score.

Churn Deep Dive:

Churn rate breakdowns by country, age group, credit score, and tenure.

Predictive Modeling:

End-to-end pipeline: one-hot encoding, class rebalancing (SMOTE+Tomek), train/test split.

Random Forest classifier with performance metrics (accuracy, confusion matrix, classification report).

Visual analytics of model scores and confusion matrix.

Model Interpretability:

LIME explainer for instance-level feature importance.

SHAP summary plots for global feature impact.

Usage
Run with Streamlit:
streamlit run xai_app.py

Upload your CSV dataset with columns matching those in the notebook.

Navigate between pages to explore data, performance, and explanations.

2. Jupyter Notebook (previously provided)
A complete, code-centric churn prediction workflow, ideal for experimentation, reproducibility, and benchmarking.

Key Workflow Steps
Data Loading: Import and preview customer churn dataset.

Exploratory Data Analysis:

Summarize data and visualize feature relationships.

Data Preprocessing:

Feature selection, encoding, normalization, and addressing class imbalance.

Modeling:

Use LazyPredict’s LazyClassifier to compare multiple algorithms.

Evaluation:

Benchmark with metrics: accuracy, precision, recall, F1 score.

Visualization of results for quick comparison.

Ideal For
Learners wanting to quickly run and modify end-to-end ML experiments.

Analysts and teams building structured ML pipelines for churn prediction.

Suggested Combined Project Structure
Filename	Purpose
README.md	Project overview and quickstart (see above)
xai_app.py	Streamlit web app for upload/EDA/modeling/XAI
notebook.ipynb	Data science workflow for exploration/modeling
How to Use
For Analysts and Experimenters:

Start with the Jupyter notebook to understand, benchmark, and tweak models.

For Exploratory/Business Users:

Use the Streamlit app for visual insights, stakeholder demos, and on-the-fly explainability.

You can seamlessly use both tools on the same CSV dataset. Adjust preprocessing and feature handling as needed to support custom data formats or additional features.
