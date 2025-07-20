# Iris Classification Project
An introductory machine learning project where I built a classifier to predict the species of an iris flower based on its physical measurements—sepal length, sepal width, petal length, and petal width.

**Overview**
The goal was to train a model that can distinguish between three iris species—Setosa, Versicolor, and Virginica. The project is based on the classic Iris dataset from UCI ML Repository.

**Tech Stack**
-Language: Python

**Libraries:**

  -pandas for data handling
  
  -matplotlib & seaborn for quick visualizations
  
  -scikit-learn for model building and evaluation
  
  -pickle for saving the trained model
  
  -Streamlit for deployment

**Key Steps**

_Data Exploration:_

  Checked class distribution and feature relationships using histograms and scatter plots.
  
  No missing values, so no data cleaning was needed.

_Preprocessing:_
  Feature scaling was done using StandardScaler.
  
_Model Training:_

Tried out several models, but Logistic Regression hit the best balance between performance and interpretability.

Tuned hyperparameters using GridSearchCV.

**Evaluation:**
  Achieved a high accuracy on the test set.
  Plotted confusion matrix to confirm consistent performance across all classes.

**Deployment:**

  Built a Streamlit app to make predictions from new inputs.

  Deployed the model using Streamlit Cloud.

_On Streamlit:_
You can test the deployed version here (https://iris-classification-vbbjjcgwne2spwumt8bqqs.streamlit.app/).


_What I Learned_
How to apply basic ML workflow from data loading to deployment

Importance of scaling and proper train-test splits

Basics of deploying ML models with Streamlit

