# Heart-Disease-Classification-DSA1101

#### Motivation
This project was completed for the NUS module DSA1101: Introduction to Data Science. The goal was to use a small medical dataset to classify whether a patient has heart disease. This was an early introduction to practical modelling in R, and it helped me understand how simple statistical methods can support medical decision making. Working with real patient variables also showed how basic exploratory analysis can guide model building.

#### Methodology

The analysis used clinical records from 300 patients, including variables such as age, cholesterol level, blood pressure, chest pain type, ECG results, and exercise-induced symptoms. The data was cleaned, summarised, and explored through histograms, boxplots, and correlation checks to identify patterns linked to heart disease and to guide feature selection.

Three models were developed in R: K-Nearest Neighbours, Decision Trees, and Logistic Regression. Each model was tuned using 5-fold cross-validation and assessed using True Positive Rate, Precision, ROC curves, and AUC. Logistic Regression delivered the strongest performance, achieving the highest AUC and the most balanced classification results across metrics.
