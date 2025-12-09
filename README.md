# Student Performance Predictor (Machine Learning + Streamlit)

This project predicts a student's final grade (G3) based on different factors like study time, absences, previous grades (G1, G2), family background, and habits like alcohol consumption and social activity.

The project uses the Student Performance Dataset from the UCI Machine Learning Repository (student-mat.csv).



The app is deployed using Streamlit and includes SHAP explainability to show how each feature affects the prediction.


## Features of the App

- Predict final grade (G3) from user inputs
- Interactive Streamlit interface
- Sliders for studytime, absences, alcohol use, parent education, G1/G2 grades, etc.
- Engineered features like:
  - study_efficiency
  - attendance_ratio
  - parent_edu_index
- Uses Random Forest Regressor model
- SHAP waterfall plot to explain why the model predicted a certain grade
- What-if scenario analysis


## Files in This Repository

- `app.py` → Streamlit application
- `rf_model.pkl` → trained Random Forest model
- `requirements.txt` → dependencies for Streamlit Cloud
- `README.md` → project documentation


## How to Run Locally

pip install -r requirements.txt
streamlit run app.py

yaml
Copy code


## Machine Learning Pipeline Used

1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Train-test split  
5. Scaling (for Linear Regression)  
6. Modeling  
   - Linear Regression  
   - Random Forest Regressor  
7. Model Evaluation (MAE, RMSE, R²)  
8. SHAP Explainability  
9. What-if scenario simulation  
10. Streamlit Deployment  


## Dataset

Student Performance Data Set  
UCI Machine Learning Repository  
(student-mat.csv)


## Notes

- This project is meant for educational and demo purposes.
- Predictions are based on historical data and may not generalize perfectly.
- SHAP values are used for transparency and interpretability.


## Author

Built as part of a full ML beginner-friendly guide with deployment using Streamlit.
