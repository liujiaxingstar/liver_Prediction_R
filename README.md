# Liver Disease Prediction: Imputation and Classification Benchmark

This repository contains R code for a comprehensive comparative study of machine learning models for liver disease prediction using the Indian Liver Patient Dataset (ILPD). The work evaluates multiple imputation strategies (MissForest, MICE, Bagging, KNN, CART, Mean/Median, and a novel Wasserstein-based adaptive MissForest) and classification algorithms (KNN, C5.0, Naive Bayes, Random Forest, SVM, Neural Network, etc.), with hyperparameter optimization via a hybrid DE-SSO algorithm.

## Key Features
- Missing data simulation (1%–20% missing rates)
- 8 imputation methods compared using MAE, MSE, RMSE, MAPE
- 10 classifiers benchmarked (accuracy, Kappa)
- Hybrid differential evolution + social spider optimization (DE-SSO) for random forest hyperparameter tuning
- Fully reproducible R scripts

## Dataset
- Indian Liver Patient Dataset (ILPD)
  Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Indian+Liver+Patient+Dataset)  
  - 583 samples, 10 features + 1 binary label (liver disease or not)
  - Features: Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, Alamine Aminotransferase, Aspartate Aminotransferase, Total Proteins, Albumin, Albumin/Globulin Ratio

## Repository Structure

## Requirements
- R (≥ 4.0)
- Required packages:
  ```r
  install.packages(c("randomForest", "caret", "DEoptim", "class", "C50",
                     "e1071", "neuralnet", "plotly", "gmodels", "cluster",
                     "fpc", "caTools", "ggplot2", "reshape", "GGally"))
  
