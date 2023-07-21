# Microsoft Stock Trend Prediction: Deep Learning Project Overview 
* Built a model to predict the stock prices of Microsoft Corporation.
* Trained the model on data for 10 years from 19/07/2013 to 19/07/2023 based on the NASDAQ historical data from yahoo finance.
* Performed Time Series Analysis on the closing stock prices per day as a target variable.  
* Used LSTM RNN layers to build the model . 

## Code and Resources Used 
![Python Version](https://img.shields.io/badge/Python-3.11.3-blue.svg) [![Numpy Version](https://img.shields.io/badge/Numpy-1.24.3-darkgreen.svg)](https://numpy.org/doc/stable/release/1.24.3-notes.html)
[![Pandas Version](https://img.shields.io/badge/Pandas-1.5.3-E77200.svg)](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.5.3.html) <br>
**Packages:** pandas, numpy, sklearn, matplotlib, tensorflow, keras <br>
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Dataset:**  Download the entire dataset from `MSFT.csv`


## Data Analysis
*	Dropped Adj Close column 
*	Plotted graph for closing price of stocks
*	Calculated the moving average (unweighted mean) of stocks for 100 and 200 days using the pandas rolling function
*	Visualized the trend chart of the moving averages
  

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/salary_by_job_title.PNG "Salary by Position")
![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/positions_by_state.png "Job Opportunities by State")
![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/correlation_visual.png "Correlations")


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
