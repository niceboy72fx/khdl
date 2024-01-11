import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tạo dữ liệu mô phỏng
np.random.seed(42)
num_samples = 100
food_intake = np.random.uniform(1500, 3000, num_samples)
physical_activity = np.random.uniform(0, 2, num_samples)
weight = 2000 + 3 * food_intake - 1000 * physical_activity + np.random.normal(0, 500, num_samples)

# Tạo DataFrame từ dữ liệu mô phỏng
data = pd.DataFrame({'Food Intake': food_intake, 'Physical Activity': physical_activity, 'Weight': weight})

# Chia dữ liệu thành features (X) và target (y)
X = data[['Food Intake', 'Physical Activity']]
y = data['Weight']

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Tạo lưới dữ liệu cho mặt phẳng hồi quy
food_intake_surface, physical_activity_surface = np.meshgrid(np.linspace(X_test['Food Intake'].min(), X_test['Food Intake'].max(), 100),
                                                             np.linspace(X_test['Physical Activity'].min(), X_test['Physical Activity'].max(), 100))
weight_surface = model.intercept_ + model.coef_[0] * food_intake_surface + model.coef_[1] * physical_activity_surface

# Dự đoán cân nặng cho một người mới (lượng thức ăn = 2500, lượng vận động = 1.5)
new_data_point = np.array([[2500, 1.5]])
predicted_weight = model.predict(new_data_point)

# Trực quan hóa kết quả bằng biểu đồ fit (plot)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Vẽ dữ liệu thực tế
ax.plot(X_test['Food Intake'], X_test['Physical Activity'], y_test, 'bo', label='Actual Data')

# Vẽ dữ liệu dự đoán cho điểm mới
ax.scatter(new_data_point[0, 0], new_data_point[0, 1], predicted_weight, c='r', marker='X', s=100, label='New Data Point (Prediction)')

# Vẽ mặt phẳng hồi quy dưới dạng plot
ax.plot_surface(food_intake_surface, physical_activity_surface, weight_surface, alpha=0.5, color='r', label='Regression Plane')

ax.set_xlabel('thuc an')
ax.set_ylabel('van dong')
ax.set_zlabel('can nang')
ax.set_title(' mối quan hệ giữa lượng thức ăn (x1), lượng vận động (x2) và cân nặng (y) ')
ax.legend()
plt.show()
