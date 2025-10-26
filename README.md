# Heart-disease-prediction

📘 Project Overview

This project aims to predict the likelihood of heart disease based on various health-related attributes using the Naive Bayes classification algorithm.
It uses machine learning techniques to analyze medical data and classify whether a person is at risk of heart disease or not.

#🧠 Key Steps Involved
#1. Data Collection

The dataset (Heart_Disease_Prediction.csv.xlsx) contains patient health information such as age, sex, blood pressure, cholesterol level, etc.

The data is loaded using Pandas for analysis.
df = pd.read_excel("D:/Heart_Disease_Prediction.csv.xlsx")
print(df.head())

#2. Data Preprocessing

Used Label Encoding to convert categorical variables into numeric form.

Prepared the dataset by separating features (X) and target (Y).

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Chest pain type'] = le.fit_transform(df['Chest pain type'])
df['Exercise angina'] = le.fit_transform(df['Exercise angina'])
df['Slope of ST'] = le.fit_transform(df['Slope of ST'])

#3. Data Splitting

The dataset is divided into training and testing sets using an 80:20 ratio.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

#3. Data Splitting

The dataset is divided into training and testing sets using an 80:20 ratio.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

#5. Model Evaluation

Predicted outcomes are compared with actual results using accuracy_score.

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))

#6. Prediction

The model can predict heart disease risk for new patient data.

sample_input = [[29, 0, 2, 100, 106, 1, 2, 80, 1, 1.0, 1, 0, 2]]
sample_df = pd.DataFrame(sample_input, columns=x.columns)
prediction = NB.predict(sample_df)

#Output Example
🩺 Prediction Result:
The patient has Heart Disease. Please consult a doctor.

#🧩 Technologies Used

Python

Pandas

Scikit-learn

NumPy

#📊 Algorithm Used

Naive Bayes Classifier (GaussianNB)
The Naive Bayes model is chosen because it is simple, efficient, and works well with categorical data and medical datasets.

#📈 Performance Metrics

Accuracy Score

Precision & Recall (optional for extension)

Confusion Matrix (can be added for visualization)
#Future Enhancements

Add data visualization (correlation heatmap, feature importance)

Compare with other models like Logistic Regression, SVM, or Random Forest

Build a Streamlit web app for real-time prediction
