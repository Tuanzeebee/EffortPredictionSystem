# Effort Prediction System
# Công cụ Ước tính Nỗ lực Phát triển Phần mềm

## Mô tả
Dự án này cung cấp một công cụ dựa trên web để ước tính nỗ lực phát triển phần mềm sử dụng các mô hình Machine Learning khác nhau và mô hình COCOMO II. Ứng dụng được xây dựng bằng Streamlit, cho phép người dùng nhập các thông số dự án và nhận được ước tính nỗ lực, thời gian phát triển, và quy mô nhóm dự kiến.

## Mục lục
- [Tính năng](#tính-năng)
- [Cấu trúc Dự án](#cấu-trúc-dự-án)
- [Cài đặt và Thiết lập](#cài-đặt-và-thiết-lập)
- [Dữ liệu](#dữ-liệu)
- [Quy trình Làm việc](#quy-trình-làm-việc)
  - [1. Tiền xử lý Dữ liệu](#1-tiền-xử-lý-dữ-liệu)
  - [2. Huấn luyện Mô hình](#2-huấn-luyện-mô-hình)
  - [3. Chạy Ứng dụng Streamlit](#3-chạy-ứng-dụng-streamlit)
- [Các Tệp Tạo tác Chính](#các-tệp-tạo-tác-chính)
- [Góp ý và Phát triển Thêm (Tùy chọn)](#góp-ý-và-phát-triển-thêm-tùy-chọn)

## Tính năng
- Giao diện web tương tác để nhập thông tin dự án và xem kết quả ước tính.
- Hỗ trợ nhiều chỉ số kích thước đầu vào: LOC (Lines of Code), FP (Function Points), UCP (Use Case Points).
- Cung cấp ước tính từ nhiều mô hình khác nhau:
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - MLP Regressor
  - COCOMO II
- Tự động lọc danh sách Ngôn ngữ Lập trình Chính dựa trên Loại Ngôn ngữ được chọn.
- Cập nhật kết quả ước tính theo thời gian thực khi người dùng thay đổi các giá trị đầu vào.
- Hiển thị biểu đồ so sánh nỗ lực ước tính giữa các mô hình.

## Cấu trúc Dự án
Dưới đây là một cấu trúc thư mục gợi ý. Bạn có thể cần điều chỉnh cho phù hợp với dự án của mình.
ten-du-an/
│
├── data/                     # Thư mục chứa dữ liệu thô và đã xử lý (gợi ý)
│   ├── du_lieu_tho.csv       # Ví dụ: tệp dữ liệu thô
│   └── du_lieu_da_xu_ly.csv  # Ví dụ: tệp dữ liệu sau khi tiền xử lý
│
├── notebooks/                # Các Jupyter Notebooks cho khám phá, tiền xử lý, huấn luyện (gợi ý)
│   ├── 1_tien_xu_ly_du_lieu.ipynb
│   └── 2_huan_luyen_mo_hinh.ipynb
│
├── AppPredictionEffort.py                    # Tệp chính của ứng dụng Streamlit (đổi tên nếu cần)
│
├── model/preprocessor.joblib       # Tệp preprocessor đã lưu (kết quả của quá trình tiền xử lý)
├── model/feature_names.joblib      # Tệp chứa tên các đặc trưng đã lưu (kết quả của tiền xử lý)
│
├── model/lasso_regression_model.joblib # Model Lasso đã lưu
├── model/decision_tree_model.joblib  # Model Decision Tree đã lưu
├── model/random_forest_model.joblib # Model Random Forest đã lưu
├── model/xgboost_model.joblib        # Model XGBoost đã lưu
├── model/mlp_regressor_model.joblib  # Model MLP Regressor đã lưu
│
└── requirements.txt          # Các thư viện Python cần thiết
└── README.md                 # Chính là tệp này


## Cài đặt và Thiết lập

### Yêu cầu Chung
- Python 3.8 trở lên
- pip ( trình quản lý gói của Python)
- Git (để sao chép repository, nếu có)

### Các bước Cài đặt
1.  **Sao chép Repository (nếu bạn lưu trữ trên Git):**
    ```bash
    git clone <URL-repository-cua-ban>
    cd <ten-thu-muc-du-an>
    ```

2.  **Tạo và kích hoạt môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv .venv
    # Trên Windows
    .venv\Scripts\activate
    # Trên macOS/Linux
    source .venv/bin/activate
    ```

3.  **Cài đặt các thư viện phụ thuộc:**
    Đảm bảo bạn có tệp `requirements.txt` trong thư mục gốc của dự án. Nếu chưa có, bạn có thể tạo từ môi trường đang hoạt động của mình bằng lệnh: `pip freeze > requirements.txt`.
    Sau đó, chạy lệnh sau để cài đặt:
    ```bash
    pip install -r requirements.txt
    ```
    Các thư viện quan trọng có thể bao gồm: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `xgboost`.

## Dữ liệu
-   **Dữ liệu Thô**: Đặt tệp dữ liệu thô của bạn (ví dụ: `du_lieu_tho.csv`) vào thư mục `data/` (hoặc cập nhật đường dẫn trong script tiền xử lý của bạn).
-   **Dữ liệu Đã Xử lý**: Bước tiền xử lý sẽ tạo ra một phiên bản dữ liệu đã được làm sạch, có thể cũng được lưu trong thư mục `data/`.

## Quy trình Làm việc
Quy trình làm việc của dự án bao gồm ba giai đoạn chính: Tiền xử lý Dữ liệu, Huấn luyện Mô hình, và Chạy Ứng dụng Streamlit. Hai giai đoạn đầu thường được thực hiện một lần hoặc khi cần cập nhật dữ liệu/mô hình, và chúng tạo ra các tệp tạo tác (artifact files) mà ứng dụng Streamlit sử dụng.

### 1. Tiền xử lý Dữ liệu
Bước này bao gồm việc làm sạch dữ liệu thô, thực hiện feature engineering, mã hóa các biến hạng mục (categorical variables), và chuẩn hóa/co giãn các biến số (numerical features). Kết quả của giai đoạn này là một bộ dữ liệu đã làm sạch, một đối tượng preprocessor đã được lưu (`preprocessor.joblib`), và danh sách tên các đặc trưng (`feature_names.joblib`).

-   **Script**: Giả sử bạn có một script hoặc Jupyter Notebook cho việc này, ví dụ: `notebooks/1_tien_xu_ly_du_lieu.ipynb` hoặc một tệp `tien_xu_ly.py`.
-   **Đầu vào**: Bộ dữ liệu thô (ví dụ: `data/du_lieu_tho.csv`).
-   **Đầu ra Chính**:
    -   `preprocessor.joblib`: Đối tượng ColumnTransformer (hoặc pipeline) của scikit-learn đã được huấn luyện (fit) trên dữ liệu huấn luyện. Tệp này rất quan trọng để biến đổi dữ liệu đầu vào mới trong ứng dụng Streamlit một cách nhất quán.
    -   `feature_names.joblib`: Một danh sách tên các đặc trưng sau khi tiền xử lý (ví dụ, sau khi one-hot encoding).
    -   Tệp dữ liệu đã xử lý (ví dụ: `data/du_lieu_da_xu_ly.csv`) để sử dụng cho việc huấn luyện mô hình.
-   **Cách chạy**: Thực thi script/notebook tiền xử lý của bạn. Đảm bảo các tệp `.joblib` đầu ra được lưu vào thư mục gốc của ứng dụng Streamlit (theo cài đặt `OUTPUT_DIR = "."` trong `app.py`) hoặc cập nhật các biến `PREPROCESSOR_PATH` và `FEATURES_PATH` trong `app.py`.

### 2. Huấn luyện Mô hình
Sau khi tiền xử lý dữ liệu, các mô hình Machine Learning khác nhau sẽ được huấn luyện để dự đoán nỗ lực phần mềm.

-   **Script**: Giả sử bạn có một script hoặc Jupyter Notebook cho việc này, ví dụ: `notebooks/2_huan_luyen_mo_hinh.ipynb` hoặc một tệp `huan_luyen.py`.
-   **Đầu vào**: Bộ dữ liệu đã xử lý (ví dụ: `data/du_lieu_da_xu_ly.csv`).
-   **Đầu ra Chính**: Các tệp mô hình đã được huấn luyện và lưu lại:
    -   `lasso_regression_model.joblib`
    -   `decision_tree_model.joblib`
    -   `random_forest_model.joblib`
    -   `xgboost_model.joblib`
    -   `mlp_regressor_model.joblib`
-   **Cách chạy**: Thực thi script/notebook huấn luyện mô hình của bạn. Đảm bảo các tệp model `.joblib` đầu ra được lưu vào thư mục gốc của ứng dụng Streamlit (theo cài đặt `OUTPUT_DIR = "."` trong `app.py`) hoặc cập nhật biến `MODEL_PATHS` trong `app.py`.

### 3. Chạy Ứng dụng Streamlit
Khi preprocessor, tên đặc trưng, và các mô hình đã huấn luyện được lưu ở đúng vị trí (hiện tại là cùng thư mục với `app.py`), bạn có thể chạy ứng dụng Streamlit.

-   **Lệnh chạy**:
    ```bash
    streamlit run app.py
    ```
    (Thay `app.py` bằng tên thực tế của tệp Streamlit script của bạn nếu khác.)
-   Ứng dụng sẽ tải các tệp tạo tác và cung cấp một giao diện tương tác để ước tính nỗ lực. Các trường nhập liệu nằm ở phía trên, và kết quả (bảng, biểu đồ) được hiển thị bên dưới, cập nhật theo thời gian thực khi đầu vào thay đổi.

## Các Tệp Tạo tác Chính
Các bước Tiền xử lý Dữ liệu và Huấn luyện Mô hình phải tạo ra các tệp sau trong thư mục gốc (hoặc theo chỉ định của `OUTPUT_DIR = "."` trong `app.py`) để ứng dụng Streamlit hoạt động chính xác:
-   `preprocessor.joblib`
-   `feature_names.joblib`
-   `lasso_regression_model.joblib`
-   `decision_tree_model.joblib`
-   `random_forest_model.joblib`
-   `xgboost_model.joblib`
-   `mlp_regressor_model.joblib`

## Góp ý và Phát triển Thêm (Tùy chọn)
-   Cho phép tải lên bộ dữ liệu mới để dự đoán hàng loạt.
-   Tích hợp các kỹ thuật feature engineering nâng cao hơn.
-   Thêm tính năng xác thực người dùng và lưu trữ dự án.
-   Ghi log lỗi chi tiết hơn.

Lưu ý quan trọng cho bạn:

Thay thế các thông tin giữ chỗ: Hãy thay thế <URL-repository-cua-ban>, <ten-thu-muc-du-an>, du_lieu_tho.csv, 1_tien_xu_ly_du_lieu.ipynb, 2_huan_luyen_mo_hinh.ipynb, và app.py (nếu tên tệp của bạn khác) bằng các tên và đường dẫn thực tế của dự án.
Tệp requirements.txt: Đây là tệp rất quan trọng. Nếu bạn chưa có, hãy tạo nó từ môi trường Python mà bạn đang phát triển dự án bằng lệnh pip freeze > requirements.txt và thêm nó vào repository.
Đường dẫn tệp: Đảm bảo rằng các đường dẫn đến tệp preprocessor, models, features trong code Streamlit (app.py) của bạn khớp với vị trí bạn lưu chúng sau khi chạy các bước tiền xử lý và huấn luyện. Hiện tại, code đang mong đợi chúng ở cùng thư mục với app.py (OUTPUT_DIR = ".").
Chúc bạn cập nhật dự án lên GitHub thành công!
