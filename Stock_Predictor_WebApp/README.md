# Stock Trend Predictor: Project Overview
![Deep Learning Project](https://img.shields.io/badge/Project-Deep%20Learning-blueviolet?logo=python)
![Yahoo Finance Dataset](https://img.shields.io/badge/Yahoo%20Finance-Dataset-orange?logo=yahoo)

* Built a model to predict the stock trend prices for any stock.
* Trained the model on data for 13 years from 21/07/2010 to 21/07/2023 based on the historical data from yahoo finance.
* Performed Time Series Analysis on the closing stock prices per day as a target variable.  
* Used LSTM RNN layers to build the model.
* Built a web app for the deployment of the model using streamlit app framework.

## Code and Resources Used 
![Python Badge](https://img.shields.io/badge/Python-3.8.17-black?logo=python)
![NumPy Badge](https://img.shields.io/badge/NumPy-1.19.5-darkgreen?logo=numpy)
![Pandas Badge](https://img.shields.io/badge/Pandas-1.3.4-E77200?logo=pandas) <br>
![Libraries Badge](https://img.shields.io/badge/Libraries-NumPy|Pandas|Matplotlib|Sklearn|tensorflow|keras|yfinance|streamlit-brown?logo=python) <br>
[![Article](https://img.shields.io/badge/Article-Read%20Here-brightgreen)](https://thecleverprogrammer.com/2021/05/01/real-time-stock-price-data-visualization-using-python/)
[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Here-darkred?logo=youtube)](https://www.youtube.com/watch?v=s3CnE2tqQdo&t=608s)<br>

**For Web Framework Requirements:**  ```pip install -r requirements.txt```  


## Data Analysis
*	Dropped Adj Close column
*	Dropped the original row indics and replaced it with integer indices
*	Plotted graph for closing price of stocks
*	Calculated the moving average (unweighted mean) of stocks for 100 and 200 days using the pandas rolling function
*	Visualized the trend chart of the moving averages

  
## Deployed Model
The Final Trained stock price predictor model's  code can be downloaded from `application.py` <br>

🔴 Data from 21st July 2010 to 21st July 2023 <br>

[streamlit-application.webm](https://github.com/sughoshdeshpande7/Time_Series_Forecasting/assets/75742228/43a95cf3-d254-44ae-9975-e8900ead8c8f)


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
Mean Squared Error (MSE) | 16.15468884625719
Root Mean Squared Error (RMSE) | 4.019289594724071
Mean Absolute Error (MAE) | 3.4042120123112283
R-squared (R²) | 0.9165813291619156

- on average, the squared difference between the predicted and actual prices is 16.15
- on average, the absolute difference between the predicted and actual prices is approximately 4.01
- on average, the absolute difference between the predicted and actual prices is approximately 3.4
- approximately 91% of the variance in the prices can be explained by the model.

Overall, these metrics suggest that the model is performing reasonably well in predicting prices. The R-squared value of 0.91 indicates a good fit of the model to the data, and the low RMSE and MAE values indicate that the model's predictions are relatively close to the actual prices on average.

The Final Trained microsoft stock price predictor model can be downloaded from `Trained_model.h5`
