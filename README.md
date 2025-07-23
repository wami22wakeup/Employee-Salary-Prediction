💼 Employee Salary Classification App
This is a Streamlit web app that predicts whether an employee earns >50K or <=50K annually, using machine learning models trained on the Adult Census Income Dataset.

<p align="center"> <img src="images/model_comparison.png" alt="Model Comparison" width="70%"> </p>

Multiple model predictions: Random Forest, Decision Tree, Gradient Boosting, KNN, and Neural Network

Encoded input features

Scaled predictions

Model performance visualization

🧠 Models Used
RandomForestClassifier

DecisionTreeClassifier

GradientBoostingClassifier

KNeighborsClassifier

MLPClassifier (Neural Network)

🧪 Run Locally
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit App
bash
Copy
Edit
streamlit run streamlit_app.py
The app will open in your browser at http://localhost:8501

🔧 Requirements (requirements.txt)
txt
Copy
Edit
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
You can create this file with:
pip freeze > requirements.txt (but clean up unnecessary packages manually)
