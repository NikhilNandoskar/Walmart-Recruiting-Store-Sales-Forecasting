# Walmart-Recruiting-Store-Sales-Forecasting

Dataset Used: Walmart Recruiting - Store Sales Forecasting
Link to dataset: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/overview

Task: Predict the Weekly Sales of Department

All those are Google Colab Ipynb notebooks.

Following are the steps involved in achieving the task.
1) Data Preparation:

     Using Pandas I’m reading the .csv files and storing them as Pandas Dataframe as df_features, 
     df_stores, df_test, df_train.
     After observing the dataframes I merged the same columns of df_deatures, df_stores with 
     df_train and saved it in the variable named df_frame.
     Now there were NaN values in columns MarkDown1, MarkDown2, MarkDown3, 
     MarkDown4, MarkDown5. I replaced all the NaN values with Zero.
     There are negative values in Weekly_Sales, MarkDown2, MarkDown3 columns, I used 
     “dataframe.describe()” function of Pandas for this. As these column values can not be 
     negative, I substituted the negative values with Zero.
     From the Date column I separated out the Month, Date and Year and concatenated with 
     data_frame and dropped the Data column.
     Column IsHoliday have Boolean values so I converted them to 0 and 1. 1==IsHoliday and 
     0==NoHoliday.
     Using sklearn “LabelEncoder” I converted the categorical values of column Type to numerical
     values viz 0 (A), 1(B), 2(C).
     The above entire process is followed for test dataset as well and final test dataframe is called
     data_frame_test.
     Finally the dependable and independent features are separated out.

     For Training
     
     The dependable feature/ Ground truth values termed as “y” gets the column Weekly_Sales.
     The independent features forms input features “X”. Shape of X: (421570,17).
     Here I did feature selection using “feature_importances_”. The reason behind it is explained
     in the model section.
     After doing the feature selection now shape of X becomes (421570,12).
     For Testing
     The independent features forms input features “X_test”. Shape of X_test: (115065,12).
     Here we predict “y” and append it to the submission file.
     
2) Evaluation:

     As per the competition the evaluation matrix for this dataset is Weighted Mean Absolute 
     Error (WMAE):
     
            WMAE = 1/(Ʃw ) Ʃi=1 to n wi | yi – yprediction |
            where,
               n is the number of rows
               yprediction is the predicted sales
               yi is the actual sales
               wi are weights. w = 5 if the week is a holiday week, 1 otherwise
     
     In my code I have written a function which calculates the WAME score.
     Apart from WMAE I’m also using R2_score as an evaluating matrix. R2 indicates the 
     proportion of variance of output “y” in accordance with the input features “X”.
     R2_score = 1 -  (Ʃi=1 to n (yi – yprediction)^2 )/(Ʃi=1 to n  (yi – yhat)^2)
     where,
         n is the number of data samples
         yprediction is the predicted values
         yi is the actual values
         yhat is 1/(n) Ʃi=1 to n yi
     R2_score can be sometimes negative which is a indication that the trained model is 
     performing poorly.
     The best R2_score is 1.



3) Models Trained:

   A) Artificial Neural Network from Scratch using Numpy

   Code file name is “LineaRegression_Regularization_Dropout.ipynb”(Run this file initially).
   Then you need to run “Walmart_Comp.ipynb” file for training and testing.

   This code is the extension of the assignment provided by Dr. Andrew Ng in his Specialization Course. I modified 
   his code to perform L1/L2 Regularization, Relu/Sigmoid/Softmax Activations, Dropout, Adam 
   optimization. Since this is a regression task, I built a “2 Hidden Layer” Neural Network.
 
   In my first attempt I normalized the input data using “MinMaxScaler” and “train_test_split”
   function of sklearn. I’m splitting the data in 60/40 ratio. I tried training with L1 regularization
   and the results were not satisfactory so I used L2 regularization. I observed negative value of
   R2 scores. So, I started hypertuning various parameters like weight initialization, number of 
   layers in the Neural Net, mini-batch size, ratio of train-test split, learning rate, regularization 
   parameter. The results obtained are as follows:
   
   Training: 
   
         **R2_score: 0.0815**
         **Root Mean Squared Error: 21789.2903**
   Testing:
   
         R2_score: 0.0830
         Root Mean Squared Error: 21711.7800
         
   WMAE for without feature scaled model is 14287.3711
   
   The magnitude of the mean square error is 10^4 and our inputs are in the range of 0 to 1. 
   After looking at the output predicted values of the Neural Net I decided to normalize my 
   output feature as well using the MinMaxScaler function. Also, I did feature selection and 
   again tweaked my model. The results obtained are as follows: 
   
   Training: 
   
          R2_score: 0.1048
          Root Mean Squared Error: 0.0310
   Testing: 
   
          R2_score: 0.1051    
          Root Mean Squared Error: 0.0309

   WMAE for feature scaled model is 0.0198

   
   B) Artificial Neural Network using Keras

   Code file name is “Keras-ANN.ipynb”
   Using Keras I created a Sequential model of “2-Hidden Layer”. 

   I have tried various types of initializers, activation functions for output layer, batch sizes, 
   learning rate. Here I used Kfold cross validation of 5 for splitting the data. 
   The best result I obtained are as follows: 
   
          R2_score: 0.1597
          Root Mean Square Error: 0.0282
          
   WMAE for feature scaled model is 0.0198


   C) Scikit-learn models

   Code file name is “Scikit_model.ipynb”

   In this I have tried the following regression models:

   1) Linear Regression, Ridge, Lasso, ElasticNet

   2) KNeighbors Regressor

   3) GradientBoostingRegressor, XGB, Random Forest Regressor

   Again, Kfold cross validation of 5 for splitting the data. During training one can choose any of 
   the above models for training.
   The best result I observed was on Random Forest Regressor.
   
          R2_score: 0.9792
          Root Mean Square Error: 0.0725
          
    The WMAE is calculated on the Training Dataset as the Testing Dataset does not have ground truth labels
    
          WMAE: 0.0012
    
    This is the least WMAE I have observed out of all the trained models. Also, the R2_score is 
    the highest amongst all. 

    While training I pickled the model which is then used for doing prediction on Testing Data.

    I have created the “Submission.csv” file as per the competition instruction.
    This file is my final submission file.



