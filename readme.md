# 🩺 Disease Prediction from Medical Data (Diabetes Detection)

### 👩‍💻 Author: **Qurat-ul-Aen**

---

## 📘 Project Overview

This project predicts the likelihood of diabetes in patients based on key medical features such as **age, BMI, glucose level, blood pressure, insulin, and family history**.
It applies **Machine Learning classification algorithms** like **Logistic Regression**, **SVM**, **Random Forest**, and **XGBoost** to identify patterns and classify patients as diabetic or non-diabetic.

---

## 🧠 Objective

To build an **AI-powered disease prediction model** that helps in early diagnosis using structured medical datasets and improves decision-making for healthcare professionals.

---

## ⚙️ Algorithms Used

* 🧩 Logistic Regression
* 🧠 Support Vector Machine (SVM)
* 🌲 Random Forest Classifier
* ⚡ XGBoost

Each model’s performance is compared using **accuracy, precision, recall, and F1-score** to select the most reliable classifier.

---

## 📊 Dataset

The project uses a **Diabetes dataset** that contains features such as:

* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`
* `Outcome` (0 = Non-diabetic, 1 = Diabetic)

You can download or paste the dataset into a CSV file named **`diabetes.csv`**.

---

## 🧩 Features Used

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| Age                      | Age of the patient                               |
| Glucose                  | Plasma glucose concentration                     |
| Blood Pressure           | Diastolic blood pressure                         |
| BMI                      | Body Mass Index                                  |
| Insulin                  | Insulin level in blood                           |
| DiabetesPedigreeFunction | Family history indicator                         |
| Outcome                  | Target variable (1 = diabetic, 0 = non-diabetic) |

---

## 🧰 Technologies Used

* Python 🐍
* pandas, numpy
* scikit-learn
* xgboost
* matplotlib, seaborn (for visualization)

---

## 🚀 Steps in the Project

1. **Data Loading & Preprocessing**

   * Handle missing values
   * Normalize numerical features

2. **Feature Selection**

   * Use correlation and feature importance

3. **Model Training**

   * Train models (SVM, Logistic Regression, Random Forest, XGBoost)

4. **Evaluation**

   * Compare model performances using metrics

5. **Prediction**

   * Predict diabetes status for new patient data

---

## 📈 Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

---

## 💡 Expected Output

✅ Displays accuracy scores of all models
✅ Plots confusion matrices
✅ Shows which model performs best
✅ Predicts diabetes outcome for a new patient input

---

## 🧑‍🎓 About the Author

👋 **Qurat-ul-Aen**
💻 Python Developer & AI/ML Engineer
🤖 Building AI Agents & Workflows
🌟 Passionate about Generative AI and empowering through technology
