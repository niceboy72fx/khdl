import numpy as np
import vnstock as vs
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = vs.stock_historical_data(ticker, start_date, end_date, "1D")
    return stock_data

# Ticker symbol and date range
ticker_symbol = 'STB'
start_date = '2020-01-01'
end_date = '2024-01-11'

# Fetch historical stock data
stock_data = get_stock_data(ticker_symbol, start_date, end_date)

# Extract relevant features (open, close, low, high, volume)
features = stock_data[['open', 'close', 'low', 'high', 'volume']].values

# Extract features and target for the new data point
X_new = features[-1, [0, 1, 2, 3, 4]].reshape(1, -1)
y_new_actual = stock_data['close'].iloc[-1]

# Create and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(features[:-1, :], stock_data['close'][:-1])

# Predict the close price for the new data point
predicted_close_linear = linear_model.predict(X_new)

# Print actual and predicted close prices
print(f"Actual Close Price for the New Data Point: {y_new_actual}")
print(f"Predicted Close Price for the New Data Point: {predicted_close_linear[0]}")

# Tính toán sai số dự đoán
prediction_error = y_new_actual - predicted_close_linear[0]

# In thông tin đánh giá
print(f"Sai số dự đoán: {prediction_error}")
if prediction_error == 0:
    print("Dự đoán chính xác!")
elif prediction_error > 0:
    print(f"Dự đoán lớn hơn giá thực tế: {prediction_error}")
else:
    print(f"Dự đoán nhỏ hơn giá thực tế: {abs(prediction_error)}")

# Vẽ biểu đồ so sánh giữa giá thực và giá dự đoán
plt.figure(figsize=(12, 4))
plt.plot(stock_data.index[:-1], stock_data['close'][:-1], label='Actual Close Price', marker='o')
plt.axvline(x=stock_data.index[-1], color='r', linestyle='--', label='New Data Point')
plt.plot(stock_data.index[-1], y_new_actual, marker='o', markersize=8, label='Actual New Data Point', color='g')
plt.plot(stock_data.index[-1], predicted_close_linear[0], marker='o', markersize=8, label='Predicted New Data Point (Linear Regression)', color='b')
plt.title('du doan gia co phieu bang mo hinh linear regression)')
plt.xlabel('Ngay thang')
plt.ylabel('Gia dong cuas')
plt.legend()

# Hiển thị đánh giá trực quan trên biểu đồ
plt.text(stock_data.index[-1], y_new_actual, f"Actual: {y_new_actual}", ha='right', va='bottom', color='g', fontsize=8)
plt.text(stock_data.index[-1], predicted_close_linear[0], f"Predicted: {predicted_close_linear[0]}", ha='right', va='top', color='b', fontsize=8)

plt.show()
