from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

# Chiều cao (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# Cân nặng (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Thêm một cột 1 vào ma trận X để tính toán w0 du
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Tính toán trọng số của đường fit
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

# Hiển thị trọng số w
w0, w1 = w[0][0], w[1][0]
print('w0 = {:.4f}, w1 = {:.4f}'.format(w0, w1))

# Dự đoán cân nặng cho chiều cao 172
X_new = np.array([[1, 172]]) 
y_pred = np.dot(X_new, w)
print('Dự đoán cân nặng cho chiều cao 172 là {:.2f} kg'.format(y_pred[0][0]))

# Trực quan hóa dữ liệu và đường fit
plt.plot(X, y, 'ro', label='Dữ liệu thực tế')
plt.plot([min(X), max(X)], [w0 + min(X) * w1, w0 + max(X) * w1], color='blue', linewidth=2, label='Đường fit')
plt.scatter([172], [y_pred], color='green', marker='*', s=200, label='Dự đoán cho chiều cao 172')
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.legend()
plt.show()