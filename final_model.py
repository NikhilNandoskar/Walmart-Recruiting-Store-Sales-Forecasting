# -*- coding: utf-8 -*-
"""final_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XWbv0WnyBPUZuXjhaveexEj3SsTfGLU6
"""
"""
from google.colab import drive
drive.mount('/content/drive')

# cd /content/drive/My Drive/'Colab Notebooks'/
cd /content/drive/My Drive/Walmart_Competition

ls"""

# Commented out IPython magic to ensure Python compatibility.
# Imports
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# %matplotlib inline

# Sklearn Imports
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

"""#!pip install import_ipynb
import import_ipynb
from preprocess_data import *
_ = data_preprocess()"""

from preprocess_data import data_preprocess

X,X_test,y,wmae_feature,df_test_for_submission,min_max_y = data_preprocess()

def RF_model(X_train, y_train,X_val,y_val):
  RF = RandomForestRegressor(n_estimators=50,criterion='mse')
  RF.fit(X_train, y_train)
  score_train_acc = RF.score(X_train, y_train)
  print("**** TRAINING INFORMATION ****")
  print("Training Accuracy: ", score_train_acc)
  score_val_acc = RF.score(X_val, y_val)
  print("Validation Accuracy: ", score_val_acc) 
  #y_train_check = LR.predict(X_train)
  print("------------------------------")
  print("------------------------------")
  print("**** Metrics and Scoring ****")
  y_pred_RF = RF.predict(X_val)
  print("Y_pred_RF values",y_pred_RF.max())
  print("R2 score: ",r2_score(y_val,y_pred_RF))
  print("MSE value: ",np.sqrt(mean_squared_error(y_val,y_pred_RF)))
  
  pkl_filename = "RF_pickle_model.pkl"
  with open(pkl_filename, 'wb') as file:
    pickle.dump(RF, file)
  return score_train_acc, score_val_acc,y_pred_RF

def train_val_model(models):
  print("The model user have provided for tarining:", models)
  n_folds = 5 # Number of folds for Cross Validation
  train_acc, val_acc,y_pred_values = [], [], [] # Lists for storing accuracies on each fold

  # Cross validation
  kfold = KFold(n_splits=n_folds, shuffle=True, random_state=7)
  for train, val in kfold.split(X, y):
    X_train, X_val = X[train], X[val] # Grabbing X_train, X_val for Models
    y_train, y_val = y[train], y[val] # Grabbiing y_train, y_val for Models
   
    rf_model = RF_model(X_train, y_train,X_val,y_val)
    train_acc.append(rf_model[0])
    val_acc.append(rf_model[1])
    y_pred_values.append(rf_model[2])
    #print("TA",train_acc)
    print("**** VALIDATION INFORMATION ****")
    print("Training Accuracy During KFolds: ", sum(train_acc)/n_folds)
    print("Validation Accuracy During KFolds: ", sum(val_acc)/n_folds)
    print("Y_pred Stats",y_pred_values)

if __name__ == "__main__":
  model = "RF" # Depends on User, User can select models to experiment from the given list ["LR","LR_R","LR_L","LR_E","RF","GB","XGB","KNN"]
  _ = train_val_model(models=model)
  #for el in _: print(el)
  
  

  # Loading the trained pickled model
  pkl_filename = "RF_pickle_model.pkl"
  with open(pkl_filename, 'rb') as file:
      trained_model = pickle.load(file)

  y_predictions = trained_model.predict(X_test)

  y_predictions_train = trained_model.predict(X)

  #y_predictions_train

  def calculate_wmae(y_predictions,y_truth,wmae_feature):
      wmae = np.zeros(y_predictions.shape)
      y_truth = y_truth.reshape(-1,1)
      #print(y_predictions.shape, y_train.shape)
      for i in range(len(y_predictions)):
          if wmae_feature[i] == 1:
              wmae[i] = 5*np.absolute(y[i]-y_predictions[i])
          else:
              wmae[i] = np.absolute(y[i]-y_predictions[i])
      
      wmae_feature = np.where(wmae_feature==1,5,wmae_feature)
      wmae_feature = np.where(wmae_feature==0,1,wmae_feature)
      return np.sum(wmae)/np.sum(wmae_feature)

  print(calculate_wmae(y_predictions_train,y,wmae_feature))

  y_actual_pred = min_max_y.inverse_transform(y_predictions.reshape(-1,1))

  #y_actual_pred.max()

  X_test_for_submission = df_test_for_submission.drop("IsHoliday",axis=1).values

  ret = np.column_stack((X_test_for_submission, y_actual_pred))

  # Final Submission File
  pd.DataFrame(ret).to_csv("Submission.csv", sep = ",", header = ["Store", "Department", "Datetriplet", "Weekley_Sales"], index = False)

