# Microsoft Stock Trend Prediction: Deep Learning Project Overview
![Deep Learning Project](https://img.shields.io/badge/Project-Deep%20Learning-blueviolet?logo=python)
[![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-MSFT%20Prices-informational?logo=yahoo)](https://finance.yahoo.com/quote/MSFT)
* Built a model to predict the stock trend prices for Microsoft Corporation.
* Trained the model on data for 10 years from 19/07/2013 to 19/07/2023 based on the historical data from yahoo finance.
* Performed Time Series Analysis on the closing stock prices per day as a target variable.  
* Used LSTM RNN layers to build the model.

## Code and Resources Used 
![Python Badge](https://img.shields.io/badge/Python-3.11.3-black?logo=python)
![NumPy Badge](https://img.shields.io/badge/NumPy-1.24.3-darkgreen?logo=numpy)
![Pandas Badge](https://img.shields.io/badge/Pandas-1.5.3-E77200?logo=pandas) <br>
![Libraries Badge](https://img.shields.io/badge/Libraries-NumPy|Pandas|Matplotlib|Sklearn|tensorflow|keras-brown?logo=python) <br>

**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Dataset:**  Download the entire dataset from `MSFT.csv`


## Data Analysis
*	Dropped Adj Close column
*	Plotted graph for closing price of stocks
*	Calculated the moving average (unweighted mean) of stocks for 100 and 200 days using the pandas rolling function
*	Visualized the trend chart of the moving averages
  

## EDA
 
![alt text](https://github.com/sughoshdeshpande7/Time_Series_Forecasting/blob/c7df1b028ac27f841cd4005f449a1da8611b2299/Microsoft_Stock_Trend_Prediction/images/closing%20prices.png)
![alt text](https://github.com/sughoshdeshpande7/Time_Series_Forecasting/blob/c7df1b028ac27f841cd4005f449a1da8611b2299/Microsoft_Stock_Trend_Prediction/images/100%20days%20moving%20average.png))
![alt text](https://github.com/sughoshdeshpande7/Time_Series_Forecasting/blob/c7df1b028ac27f841cd4005f449a1da8611b2299/Microsoft_Stock_Trend_Prediction/images/200%20days%20moving%20average.png)
![alt text](https://github.com/sughoshdeshpande7/Time_Series_Forecasting/blob/c7df1b028ac27f841cd4005f449a1da8611b2299/Microsoft_Stock_Trend_Prediction/images/final%20predictions.png)


## Model Building 

Split the data into train and tests sets with a test size of 30%.<br>
Scaled down the dataset to be made usable for training model using Min Max Scaler.<br>
Transformed the training and test dataset into a numpy array before using it in the model.<br>

LSTM was used to build the model because:
  * **Sequential Data Handling** – LSTMs are designed to work very well in handling sequential data like stock prices where each data point depends on previous data points in a time series
  * **Long-Term Dependencies** - LSTM is specifically designed to address the vanishing gradient problem that affects traditional RNNs when dealing with long-term dependencies in sequential data.
  * **Memory and Forget Gates** - LSTMs have memory cells and mechanisms to control the flow of information through the cell, including the ability to forget old information and retain relevant information. This is beneficial for stock price prediction as it allows the model to adapt and learn from changing market conditions.
  * **Handling Noisy Data** - Financial time series data can be noisy and volatile, with sudden price fluctuations. LSTMs can handle noisy data and generalize patterns despite the noise, making them robust for modeling stock prices.
  * **Feature Extraction** - LSTMs can automatically learn and extract relevant features from the input data, reducing the need for manual feature engineering. This is especially advantageous when dealing with complex patterns in financial markets.
  * **Flexibility** - LSTMs can be combined with other deep learning architectures or traditional machine learning models to build more complex models for stock price prediction. They are versatile and can be integrated into larger prediction systems.

 A relu activation function is used and the dropout value increases by 0.1 for every layer<br>
 An Adam Optimizer and mean squared error loss function was used to train the model for 50 epochs.

## Model performance

Metrics       | Value
------------- | -------------
Mean Squared Error (MSE) | 121.61146822351022
Root Mean Squared Error (RMSE) | 11.027758984649157
Mean Absolute Error (MAE) | 8.785182568524617
R-squared (R²) | 0.9185184115873659

- on average, the squared difference between the predicted and actual prices is 121.61
- on average, the absolute difference between the predicted and actual prices is approximately 11.03
- on average, the absolute difference between the predicted and actual prices is approximately 8.79
- approximately 92% of the variance in the prices can be explained by the model.

Overall, these metrics suggest that the model is performing reasonably well in predicting prices. The R-squared value of 0.92 indicates a good fit of the model to the data, and the low RMSE and MAE values indicate that the model's predictions are relatively close to the actual prices on average.

The Final Trained microsoft stock price predictor model can be downloaded from `stock_predictor_model.h5`
