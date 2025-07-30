Categories Classification Depending on Textual Context
Overview:
This project focuses on the classification of product categories using textual data such as titles, descriptions, and tags. The aim is to develop an efficient model to forecast product category demand, assisting in inventory management. Various machine learning models were compared to determine the most effective approach for accurate classification.
Key Objectives:
Clean and preprocess textual data for enhanced computational efficiency.
Engineer features by combining relevant textual columns.
Build and evaluate machine learning models to classify product categories:
Top Category ID
Bottom Category ID
Color ID
Dataset:
Cleaning: Removal of irrelevant and missing data, lemmatization, and removal of special characters, stopwords, and unnecessary whitespace.
Vectorization: Textual data transformed using CountVectorizer for simplicity and word-order preservation.
Models Evaluated:
Logistic Regression (Baseline Model)
Multinomial Naïve Bayes (MultinomialNB)
Support Vector Machine (SVM)
Decision Tree
Results:
Best Performance: Logistic Regression achieved the highest accuracy, F1 score, and recall for all features, especially for the Top Category ID with fewer classes.
Challenges: Classification of Color ID due to mismatches in textual and actual color data.
Model	Accuracy (Validation)	F1 Score	Recall
Logistic Regression	0.90	0.90	0.90
MultinomialNB	0.78	0.79	0.79
SVM	0.83	0.83	0.83
Decision Tree	0.74	0.74	0.74
Future Improvements:
Optimize data sampling to reduce gaps between training and validation scores.
Develop a unified model for predicting multiple features simultaneously.
Refine data cleaning and balance the dataset for enhanced model performance.
Conclusion:
The project demonstrates that even a baseline model like Logistic Regression, when paired with well-cleaned and preprocessed data, can outperform more complex algorithms. Proper data handling significantly impacts the accuracy of text classification models.
