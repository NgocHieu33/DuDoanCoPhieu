#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cài đặt thư viện
get_ipython().system('pip install yfinance scikit-learn matplotlib pandas numpy')

# Import thư viện
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Tải dữ liệu cổ phiếu Facebook (Meta)
ticker = 'META'
data = yf.download(ticker, start='2015-01-01', end='2023-12-01')

# Chuẩn bị dữ liệu
data = data[['Close']].dropna()
data['Days'] = (data.index - data.index.min()).days

X = data[['Days']]
y = data['Close']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dự đoán tương lai
future_days = pd.DataFrame({
    'Days': np.arange(data['Days'].max() + 1, data['Days'].max() + 365 * 4 + 1)  # 2025-2028
})

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)
future_pred = model.predict(future_days)

# Tính lỗi
mse = mean_squared_error(y_test, y_pred)
print(f"MSE (Mean Squared Error): {mse:.2f}")

# Biểu đồ
future_dates = pd.date_range(start=data.index.max() + pd.Timedelta(days=1), periods=365 * 4, freq='D')
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Thực tế', color='blue')
plt.plot(future_dates, future_pred, label='Dự đoán (2025-2028)', color='red', linestyle='dashed')
plt.title(f'Dự đoán giá cổ phiếu {ticker} (2025-2028)')
plt.xlabel('Ngày')
plt.ylabel('Giá đóng cửa')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




