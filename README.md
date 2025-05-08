# EffortPredictionSystem
# Công cụ Ước tính Nỗ lực Phát triển Phần mềm

Ứng dụng Streamlit này giúp ước tính nỗ lực cần thiết cho các dự án phát triển phần mềm, sử dụng mô hình COCOMO II và các mô hình Machine Learning.

## Giới thiệu sơ lược

Công cụ cho phép bạn nhập các thông số cơ bản của dự án như kích thước (LOC, FP, UCP) và ngôn ngữ lập trình. Dựa trên đó, ứng dụng sẽ cung cấp các ước tính về:

* Nỗ lực (Person-Hours / Person-Months)
* Thời gian phát triển (Tháng)
* Quy mô nhóm

Kết quả được tổng hợp từ mô hình COCOMO II và 5 mô hình Machine Learning: Linear Regression, Decision Tree, Random Forest, XGBoost, và MLP Regressor.

## Các thư viện cần thiết

Đảm bảo bạn đã cài đặt Python (phiên bản 3.7 trở lên). Sau đó, cài đặt các thư viện sau bằng pip:

```bash
pip install notebook ,streamlit ,pandas, numpy, matplotlib ,seaborn ipykernel,  scikit-learn, matplotlib, xgboost

Sau khi cài đặt, bạn có thể khởi chạy Jupyter Notebook bằng lệnh:

Bash

jupyter notebook

chạy các cell từ tiền xử lý dữ liệu đến các mô hình meachine learning và file tổng hợp cuối để lấy các tệp mô hình

Chuẩn bị tệp mô hình
Để ứng dụng hoạt động, bạn cần có các tệp mô hình (.joblib) sau trong cùng thư mục với tệp mã nguồn Python:

preprocessor.joblib
feature_names.joblib
linear_regression_model.joblib
decision_tree_model.joblib
random_forest_model.joblib
xgboost_model.joblib
mlp_regressor_model.joblib
Đây là các tệp chứa preprocessor và các mô hình Machine Learning đã được huấn luyện.
Cách chạy ứng dụng
Mở Terminal (hoặc Command Prompt).
Di chuyển đến thư mục chứa tệp mã nguồn Python của bạn và các tệp .joblib.
Bash

cd path/to/your/streamlit_app_directory
Thực thi lệnh sau:
Bash
File web run cuối cùng của mình là FinalCodeUseAPP.py
streamlit run your_script_name.py
(Thay your_script_name.py bằng tên tệp Python của bạn).
Ứng dụng sẽ tự động mở trong trình duyệt web của bạn.

Chúc bạn sử dụng hiệu quả!
