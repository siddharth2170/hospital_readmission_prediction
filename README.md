# Hospital Readmission Prediction

This repository contains code for predicting 30-day hospital readmissions using the **Diabetes 130-US hospitals dataset**.  
The study integrates clinical and non-clinical features with advanced ML and DL models.

## Models Implemented
- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees
- Deep Neural Network (DNN)
- 1D Convolutional Neural Network (CNN)

## Results (from research)
- **DNN**: Accuracy = 0.9074, F1 = 0.8881, AUC = 0.9775
- **1D-CNN**: Accuracy = 0.8516, F1 = 0.8159, AUC = 0.9370

## How to Run
```bash
pip install -r requirements.txt
python src/train_dnn.py
```

## Dataset
Available at: [UCI Repository]([https://archive.ics.uci.edu/dataset/296/diabetes+130/us+hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008))
