import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import streamlit as st

start_date = "2010-07-21"
end_date = "2023-07-21"



st.title('Real - Time Stock Trend Predictor')

user_input = st.text_input("Enter the Company",'AAPL')
df = yf.download(user_input, start=start_date, end=end_date)

# Describing Data

st.subheader('Data from 7th July 2010 to 7th July 2023')
st.write(df.describe())

#                                     Visualizations


# First Plot: Closing Price vs. Time

fig, ax1 = plt.subplots() 
ax = df["Close"].plot(figsize=(12, 8), title=user_input+" Stock Prices vs Time", fontsize=20, label="Closing Price")
plt.legend()
plt.grid()
st.pyplot(fig)  # Display the First plot using Streamlit

# Second Plot: Closing Price and 100-day Rolling Mean Avg vs. Time
fig2, ax2 = plt.subplots(figsize=(12, 8))
ma100 = df["Close"].rolling(100).mean()
ax2.plot(df["Close"], label="Closing Price")
ax2.plot(ma100, 'r',label="100-day Rolling Mean")
ax2.set_title(user_input + " Stock Prices vs Time with 100 Moving Average", fontsize=20)
ax2.set_xlabel("Time")
ax2.set_ylabel("Closing Price")
ax2.legend()
ax2.grid()

# Display the second plot using Streamlit
st.pyplot(fig2)

# Third Plot: Closing Price and 100-day and 200 Rolling Mean Avg vs. Time
fig3, ax3 = plt.subplots(figsize=(12, 8))
ma100 = df["Close"].rolling(100).mean()
ma200 = df["Close"].rolling(200).mean()
ax3.plot(df["Close"], label="Closing Price")
ax3.plot(ma100,'r', label="100-day Rolling Mean")
ax3.plot(ma200, 'indigo',label="200-day Rolling Mean")
ax3.set_title(user_input + " Stock Prices vs Time with 100 & 200 Moving Averages", fontsize=20)
ax3.set_xlabel("Time")
ax3.set_ylabel("Closing Price")
ax3.legend()
ax3.grid()

# Display the second plot using Streamlit
st.pyplot(fig3)

# Splitting Data into 70% Train and 30% Test Set
train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(train.shape)
print(test.shape)

# Scaling down the data to 0 and 1 to be useable in an LSTM Model
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

# Convert Training Data into Array
train_arr=scaler.fit_transform(train)




# Load LSTM model

model = load_model('Trained_model.h5')



#                          Code for Testing the Model



# Extract last 100 days (rows) from training dataset
# store it in a variable to predict the value of the next day as test dataset
#                          - TIME SERIES ANALYSIS
                                
past_100_days = train.tail(100)

# Connect Last 100 days training data with the test data
final_df = past_100_days.append(test,ignore_index=True) 

input_data = scaler.fit_transform(final_df) # Perform Feature Scaling of data

x_test=[] # List to save the testing values
y_test=[] # List to save the predicted values

for i in range(100,input_data.shape[0]): # Test the data on the first 100 values
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
# Convert to numpy array to make it fit for model     
x_test , y_test = np.array(x_test),np.array(y_test) 

y_predicted=model.predict(x_test) # Making Predictions
scaler = scaler.scale_  # Find Factor by which the values have been scaled down

# Convert values to original form so that they can be plotted
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



#                                 FINAL GRAPH

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(y_test,'g',label='Original Price')
ax4.plot(y_predicted,'darkorange',label='Predicted Price')
ax4.set_title('Original vs. Predicted Stock Prices', loc='center', 
           bbox=dict(facecolor='lightgray', edgecolor='white', 
           boxstyle='round,pad=0.5',linewidth=2))
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
st.pyplot(fig4)
