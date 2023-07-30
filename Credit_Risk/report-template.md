# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* The purpose of the analysis.

The purpose of this analysis is to see the accuracy of a machine learning model that predicts the creditworthiness of borrowers with the given dataset

* Financial information the data was on, and what we needed to predict.

The given financial dataset has several feature parameters or independent variable like 'loan_size', 'interest_rate', 'borrower_income', 'debt_to_income','num_of_accounts', 'derogatory_marks', 'total_debt' and a target or dependent variable 'loan_status'. Based on the feature variables the target variable will either be 0 or 1. loan_status = 0 indicates, the loan given to the borrower with the relative feature parameter, is healthy or has very good possibility for loan repayment. If loan_Status = 1, then chances of repayment is very low, and the loan has high risk of defaulting.

Using the given dataset, we design a classification model to train the machine, test a new set of similar datasets and predict the loan status of a borrower.


* The stages of the machine learning process went through as part of this analysis.

Step 1: Data Pre-processing 
            * Creating a dataframe by reading the source data in CSV formate.
            * Seperate the data into labels (target variable) and features (independent variable) and save to variable y & X
Step 2: Prepare training and testing dataset using train_test_split
Step 3: Create Logistic Regression model by importing LogisticRegression from sklearn
Step 4: Fit the training dataset(X_train) to logistc regresion 
Step 5: After trainging, pass the testing dataset(X_test) and predict the result.
Step 6: Validating the prediction.

Repeat the above steps except, import RandomForestClassifier method from sklearn to train machine using Random Forest classifier. 

## Results

Precision and recall scores of all machine learning models.

* Machine Learning Model 1 (Logistic Regression):
          Predicted 0 Predicted 1
Actual 0  18657       108
Actual 1  52          567

* Model 1 Accuracy, Precision, and Recall scores.
                precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Precision: Out of all the loan that the model predicted 85% would get defaulted and 100% would be healthy

Recall: Out of all the loan that did get default, the model predicted this outcome correctly for 99% of those healthy loan.

F1 Score: Since this value is very close to 1, it tells us that the model does a good job of predicting whether a loan will get defaulted.

Support: These values simply tell us how many loans belonged to each class in the test dataset. We can see that among the loans in the test dataset, 18765 is in healthy status and 619 did get defaulted.

* Machine Learning Model 2 confusion matrix (Random Forest):

          Predicted 0 Predicted 1
Actual 0  18659       106
Actual 1  44          575

* Model 2 Accuracy, Precision, and Recall scores.
Accuracy Score : 0.9922616591002889
Classification Report
                precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.93      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.96      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Precision: Out of all the loan that the model predicted 85% would get defaulted and 100% would be healthy

Recall: Out of all the loan that actually did get default, the model predicted this outcome correctly for 99% of those healthy loan.

F1 Score: Since this value is very close to 1, it tells us that the model does a good job of predicting whether a loan will get defaulted.

Support: These values simply tell us how many loans belonged to each class in the test dataset. We can see that among the loans in the test dataset, 18765 is in healthy status and 619 did get defaulted.

For both the model, the score and other evaluation factors are similar. So, analysis report is same for random forest model

## Summary

Used logistic Regression model & random forest to predict the output.  The F1 score is 1 for healthy loan which is very great score, which indicates prediciting whether a loan is healthy or not would be highly accurate.At the same time F1 for predicitng whether a loan will get defaulted is 0.88, which is not a great score but still near to 1. 

The objective is to design a model that predicts the risk factor, which means predicitng the loan that has a high risk of defaulting. Though the score for predicitng healthy loan is good (1), the score for predicitng risky loan (0.88) is not great. So would reccomend to try other classification model and check if their accuracy score in predicitng risky loan is much better than the 2 model. If we couldn't find a model with better accuracy than logistic regression / random forest, then sticking to either one of the models is recommended.



