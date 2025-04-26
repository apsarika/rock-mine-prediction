# rock-mine-prediction
This project aims to classify sonar data to distinguish between rocks and mines using machine learning models. The goal is to predict whether a given sonar signal corresponds to a rock or a mine based on the features extracted from sonar waves.

##Tech Stack:
Python Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn

##Machine Learning Models:
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
LightGBM
CatBoost
HistGradientBoosting

##Model Evaluation:
Accuracy, Precision,Confusion Matrix

##Key Features:
Data Preprocessing: Clean, process, and handle missing data.

##Model Training & Comparison: Compare the performance of classic models (Logistic Regression, KNN, etc.) with newer ones (LightGBM, CatBoost).

##Evaluation: Evaluate models using precision-weighted voting and visualize the results using plots.

##How to Run Locally:
Clone the repository with the command git clone https://github.com/apsarika/rock-mine-prediction.git.
Then, navigate into the project folder by running cd rock-mine-prediction. After that, install all the required libraries by executing pip install -r requirements.txt. 
If you don't have a requirements.txt file, you can manually install dependencies like scikit-learn, pandas, and numpy.
Once the dependencies are set up, run the script using python main.py.
After running the script, you'll see the model's evaluation results in the terminal.
