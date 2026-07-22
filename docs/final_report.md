
###Credit Card Fraud Detection Evaluation Report



##Project Objective

The aim of this project was to build an end-to-end machine learning pipeline capable of detecting fraudulent credit card transactions. 

The project involved combining multiple datasets, engineering new behavioural features, training several machine learning models and selecting the best-performing model.



##Dataset

Three datasets were combined during the ETL process:

Transactions
Customer Profiles
Terminal Profiles

These were merged into a single dataset before feature engineering was applied.



##Feature Engineering

Several new features were created to improve the model's ability to identify fraudulent behaviour, including:

Transaction hour
Day of week
Is the weekend
Night transaction indicator
Amount difference from customer average
Amount ratio and z-score
Minutes since previous transaction
Terminal change indicator
Rapid terminal change
Distance between customer and terminal

These features were designed to capture unusual customer behaviour rather than relying only on the raw transaction data.



##Models Evaluated

Model	ROC-AUC
Logistic Regression	0.8965
Random Forest	0.9603
Gradient Boosting	0.9662
Gradient Boosting (Tuned)	0.9662

The tuned Gradient Boosting model achieved the highest overall performance and was selected as the recommended model.



Confusion Matrix (Gradient Boosting Tuned Model)

True Negatives: 345,186
False Positives: 1,162
False Negatives: 4,897
True Positives: 5,817


Figure 1: Confusion matrix for the tuned Gradient Boosting model. 
The model correctly classified 350,003 of 357,062 transactions, achieving an overall accuracy of approximately 98% while maintaining a ROC-AUC score of 0.9662.

The model correctly classified the vast majority of legitimate transactions while successfully detecting over 5,800 fraudulent transactions. 

Although some fraudulent transactions were missed, the overall performance was strong for this highly imbalanced classification problem.



Key Findings
Successfully combined three datasets into a single machine learning dataset.
Engineered behavioural features to improve fraud detection.
Compared multiple machine learning algorithms.
Gradient Boosting produced the best overall performance.
The final model achieved a ROC-AUC score of 0.9662, demonstrating excellent ability to distinguish fraudulent from legitimate transactions



Conclusion

This project successfully produced a complete machine learning pipeline for credit card fraud detection. 

From data extraction through to model evaluation and hyperparameter tuning, each stage of the workflow was implemented and tested. 

The final Gradient Boosting model demonstrated excellent predictive performance and provides a strong foundation for detecting potentially fraudulent transactions in a real-world setting.


