# Customer Churn Prediction Project

## Project Overview
This project implements an advanced machine learning pipeline for predicting customer churn. It includes comprehensive feature engineering, model training, and evaluation components, along with various statistical tests and feature importance measures.

## Features
- Advanced feature engineering
- Safe label encoding for categorical variables
- Ensemble modeling with Logistic Regression and Ridge Classifier
- K-fold cross-validation
- ROC-AUC evaluation and visualization
- Feature importance analysis
- Additional statistical tests and measures:
  - PhiK correlation
  - Mutual Information (MI)
  - Weight of Evidence (WoE)
  - Information Value (IV)
  - Correlation testing

## Project Structure
- `main.ipynb`: The main script that orchestrates the entire pipeline
- `SafeLabelEncoder`: Custom encoder for handling unknown labels
- `AdvancedFeatureEngineer`: Class for creating advanced features
- `OptimizedModelTrainer`: Class for training and ensembling models
- Utility functions for plotting and evaluation


#Feature Importance
![image](https://github.com/user-attachments/assets/51b2332e-202e-408c-ba9c-a3d08443db53)

#ROC Curve
![image](https://github.com/user-attachments/assets/43f60b20-5334-49fa-b3f9-7ac9653e60f5)

## How to Use
1. Ensure you have the required dependencies installed (pandas, numpy, scikit-learn, matplotlib, seaborn, scipy)
2. Place your `train.csv` and `test.csv` files in the same directory as the script
3. Run the main script:
   ```
   python main.py
   ```
4. The script will generate:
   - A submission file `submission_optimized.csv`
   - ROC curve comparison plot (`roc_curves_comparison.png`)
   - Feature importance plot (`feature_importance.png`)

## Additional Statistical Measures

### PhiK Correlation
PhiK correlation is used to measure the correlation between categorical variables or between categorical and continuous variables. It's particularly useful in this project for understanding relationships between categorical features and the churn target.

### Mutual Information (MI)
Mutual Information is calculated to measure the mutual dependence between features and the target variable. It helps in feature selection by identifying the most informative features for predicting churn.

### Weight of Evidence (WoE) and Information Value (IV)
WoE and IV are used to assess the predictive power of individual features. They're particularly useful for binary classification problems like churn prediction:
- WoE measures how much the odds of an outcome (churn) change based on a particular value of a feature.
- IV summarizes the predictive power of a feature across all its values.

### Correlation Testing
Various correlation tests are performed to understand the relationships between features and their impact on churn. This includes Pearson correlation for continuous variables and chi-square tests for categorical variables.

## Results
The project outputs several key results:
1. ROC-AUC scores for individual models and the ensemble
2. ROC curve comparison plot
3. Feature importance rankings
4. Statistical test results for feature selection and importance

## Future Improvements
- Implement more advanced models (gradient boosting, neural networks)
- Add more sophisticated feature selection techniques
- Integrate explainable AI techniques for model interpretation

## Contributors
[O'ktambek Amonov]
