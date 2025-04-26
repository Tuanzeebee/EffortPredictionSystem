import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib

matplotlib.use('Agg')  # Đặt backend thành Agg trước khi import pyplot
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

app = Flask(__name__)

# Dữ liệu mẫu
data = {
    'LOC': [1000, 2000, 5000, 8000, 10000, 15000, 20000, 30000, 40000, 50000],
    'FP': [50, 100, 200, 320, 400, 600, 800, 1200, 1600, 2000],
    'UCP': [10, 20, 50, 80, 100, 150, 200, 300, 400, 500],
    'Effort': [20, 40, 100, 160, 200, 300, 400,600, 800, 1000]
}
df = pd.DataFrame(data)

# Tiền xử lý dữ liệu
X = df[['LOC', 'FP', 'UCP']]
y = df['Effort']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)


# Hàm COCOMO II
def cocomo_ii_effort(size, a=2.5, b=0.8, effort_multipliers=0.9):
    return a * (size ** b) * effort_multipliers


# Đánh giá mô hình
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")


print("Đánh giá mô hình:")
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
nn_pred = nn_model.predict(X_test)
cocomo_pred = [cocomo_ii_effort(size) for size in X_test['LOC']]
evaluate_model(y_test, lr_pred, "Linear Regression")
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, nn_pred, "Neural Network")
evaluate_model(y_test, cocomo_pred, "COCOMO II")


# Flask routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        loc = float(request.form['loc'])
        fp = float(request.form['fp'])
        ucp = float(request.form['ucp'])

        input_data = pd.DataFrame([[loc, fp, ucp]], columns=['LOC', 'FP', 'UCP'])
        lr_effort = lr_model.predict(input_data)[0]
        rf_effort = rf_model.predict(input_data)[0]
        nn_effort = nn_model.predict(input_data)[0]
        cocomo_effort = cocomo_ii_effort(loc)

        # Tạo biểu đồ cột
        models = ['Linear Regression', 'Random Forest', 'Neural Network', 'COCOMO II']
        efforts = [lr_effort, rf_effort, nn_effort, cocomo_effort]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, efforts, color=['blue', 'green', 'orange', 'red'])
        plt.title('Dự đoán Effort (người-tháng)')
        plt.ylabel('Effort (người-tháng)')

        # Thêm số liệu trên cột
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05 * yval, f'{yval:.2f}',
                     ha='center', va='bottom')

        # Lưu biểu đồ
        plt.savefig('static/effort_chart.png')
        plt.close()  # Đóng để tránh giữ trong bộ nhớ

        return render_template('result.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)