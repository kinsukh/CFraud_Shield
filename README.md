# Credit Card Fraud Detection

This repository contains a project focused on detecting credit card fraud using machine learning techniques. The goal is to develop a reliable model that can distinguish between legitimate and fraudulent transactions, addressing challenges such as data imbalance and ensuring accurate predictions.

## Project Overview

Credit card fraud detection is a critical area of focus in the financial industry. This project uses a dataset with various transaction features to identify patterns associated with fraud. The project workflow includes data preprocessing, exploratory data analysis, model training, and evaluation.

## Dataset

The dataset used in this project includes the following columns:
- **V1 to V28:** Unknown features representing principal components obtained from PCA.
- **Time:** The time elapsed between the transaction and the first transaction in the dataset.
- **Amount:** The transaction amount.
- **Class:** The target variable where `0` indicates a legitimate transaction and `1` indicates a fraudulent transaction.

### Key Insights:
- The dataset is highly imbalanced, with a majority of transactions being legitimate.
- The `Class` variable serves as the target for model predictions.
- Fraudulent transactions are typically associated with lower amounts.

## Exploratory Data Analysis

During the exploratory data analysis, the following observations were made:
- The dataset's imbalanced nature could lead to overfitting if not addressed.
- Fraudulent transactions generally involve smaller amounts, often below 2500.

## Modeling Approach

The project involves the following steps:
1. **Data Preprocessing:** Handling missing values, feature scaling, and addressing class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).
2. **Model Training:** Various machine learning models such as Logistic Regression, Decision Trees, and Random Forests are trained on the dataset.
3. **Model Evaluation:** Models are evaluated based on metrics like accuracy, precision, recall, and F1-score to ensure balanced performance across both classes.

## Visualizations

The project includes several visualizations to aid in understanding the data and model performance:
- **Count Plots:** Show the distribution of legitimate and fraudulent transactions.
- **Scatter Plots:** Highlight the relationship between transaction amount and fraud likelihood.
- **Confusion Matrix:** Used to evaluate model predictions.

## Getting Started

### Prerequisites

Ensure you have the following software installed:
- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `plotly`, `imbalanced-learn`

### Installation

#### Clone this repository to your local machine:

```
git clone https://github.com/kinsukh/credit-card-fraud-detection.git
```
#### Navigate to the project directory:

``` 
cd credit-card-fraud-detection
 ```

#### Install the required dependencies:

``` 
pip install -r requirements.txt
 ```

#### Usage
To explore the analysis and run the model, open the Jupyter Notebook:

```
jupyter notebook final_report_model.ipynb
 ```

Execute the cells to perform data analysis, model training, and evaluation.

### Conclusion
- The conclusion is that Logistic Regression is the most reliable model for detecting fraudulent transactions in this context, outperforming KNN and Random Forest Classifiers based on key classification metrics.
- Fraudulent transactions are consistently below an amount of 2500, suggesting that fraudsters intentionally target smaller amounts to evade detection.

### Results
- The final model shows effective detection of fraudulent transactions, with balanced performance across both legitimate and fraudulent classes. The use of SMOTE helped in mitigating the effects of data imbalance.


