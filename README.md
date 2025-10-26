#🫀 HEART DISEASE PREDICTION USING NAIVE BAYES CLASSIFIER
===========================================================

📘 Project Overview:
This Python project predicts whether a person is likely to have 
heart disease based on their health attributes using a 
Gaussian Naive Bayes (GNB) Machine Learning model.

It involves reading a medical dataset, preprocessing data, 
training a model, and making predictions.

-----------------------------------------------------------
🧠 STEPS INVOLVED:
-----------------------------------------------------------

1️⃣ Data Collection:
   - The dataset 'Heart_Disease_Prediction.csv.xlsx' is loaded using pandas.
   - It contains medical details such as age, sex, cholesterol, blood pressure, etc.

2️⃣ Data Preprocessing:
   - Label Encoding is applied to convert categorical values into numeric form.
   - Features (X) and target (Y) are separated for training.

3️⃣ Data Splitting:
   - The dataset is divided into 80% training and 20% testing using train_test_split.

4️⃣ Model Training:
   - Gaussian Naive Bayes (from sklearn) is used to train the model on the training data.

5️⃣ Model Evaluation:
   - Predictions are compared with actual test data.
   - Accuracy of the model is calculated using accuracy_score.

6️⃣ Model Prediction:
   - A sample input can be tested to check if the patient is at risk of heart disease.

-----------------------------------------------------------
🧩 TECHNOLOGIES USED:
-----------------------------------------------------------
   - Python
   - Pandas
   - Scikit-learn
   - NumPy

-----------------------------------------------------------
🎯 OUTPUT EXAMPLE:
-----------------------------------------------------------
🩺 Prediction Result:
The patient has Heart Disease. Please consult a doctor.

OR

🩺 Prediction Result:
The patient is Normal and healthy.

-----------------------------------------------------------
📈 FUTURE ENHANCEMENTS:
-----------------------------------------------------------
   - Add visualizations (heatmaps, correlation plots)
   - Compare with Logistic Regression or Random Forest
   - Create a Streamlit web app for real-time predictions



