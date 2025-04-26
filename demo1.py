# -*- coding: utf-8 -*-
"""
app.py: Ứng dụng Web Streamlit để dự đoán Effort
(Bao gồm Mô hình ML, COCOMO II Basic, FP, UCP và So sánh).

(Phiên bản đã sửa lỗi "Unknown Categories", cập nhật đường dẫn file,
và sửa lỗi hiển thị biểu đồ)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import OrderedDict
import math # Cần cho COCOMO
import traceback # Thêm để in lỗi chi tiết

# Import các lớp cần thiết (Giữ nguyên như cũ)
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder # Hoặc StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
except ImportError as e:
    st.error(f"Lỗi Import thư viện: {e}. Hãy đảm bảo các thư viện cần thiết (streamlit, pandas, scikit-learn, joblib, xgboost) đã được cài đặt trong môi trường của bạn.")
    st.stop()


# --- Cấu hình Trang và Tải Artifacts (Giữ nguyên như cũ) ---

st.set_page_config(page_title="So sánh Ước tính Effort Phần mềm", layout="wide")

st.title("Ứng dụng So sánh Ước tính Effort Phần mềm 📊")
st.write("""
Nhập thông tin dự án để nhận ước tính effort (person-hours) từ nhiều mô hình Machine Learning
và các phương pháp truyền thống (COCOMO II Basic, Function Points, Use Case Points).
""")

# Định nghĩa đường dẫn (Giữ nguyên)
OUTPUT_DIR = "."
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.joblib")
MODEL_PATHS = OrderedDict([
    ('Linear Regression', os.path.join(OUTPUT_DIR, "linear_regression_model.joblib")),
    ('Decision Tree', os.path.join(OUTPUT_DIR, "decision_tree_model.joblib")),
    ('Random Forest', os.path.join(OUTPUT_DIR, "random_forest_model.joblib")),
    ('XGBoost', os.path.join(OUTPUT_DIR, "xgboost_model.joblib")),
    ('MLP Regressor', os.path.join(OUTPUT_DIR, "mlp_regressor_model.joblib"))
])

@st.cache_resource
def load_all_artifacts(preprocessor_path, features_path, model_paths_dict):
    """
    Tải preprocessor, feature names, các mô hình ML, và trích xuất các danh mục.
    (Giữ nguyên logic hàm này)
    """
    loaded_models = OrderedDict()
    preprocessor = None
    feature_names = None
    categorical_features_options = {}
    original_cols_order = []
    all_loaded_successfully = True

    # --- Tải Preprocessor ---
    if not os.path.exists(preprocessor_path):
        st.error(f"LỖI: Không tìm thấy file preprocessor tại '{preprocessor_path}'")
        return None, None, None, None, None
    try:
        preprocessor = joblib.load(preprocessor_path)
        print("Tải preprocessor thành công.")
        # --- Trích xuất thông tin từ Preprocessor ---
        try:
            num_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'num')
            cat_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'cat')
            original_num_features = list(num_transformer_tuple[2])
            original_cat_features = list(cat_transformer_tuple[2])
            original_cols_order = original_num_features + original_cat_features
            print("Thứ tự cột gốc mong đợi:", original_cols_order)

            cat_pipeline = preprocessor.named_transformers_['cat']
            onehot_encoder = cat_pipeline.named_steps['onehot']

            if hasattr(onehot_encoder, 'categories_'):
                if len(onehot_encoder.categories_) == len(original_cat_features):
                    for i, feature_name in enumerate(original_cat_features):
                        categories = onehot_encoder.categories_[i]
                        categorical_features_options[feature_name] = categories.tolist()
                    print("Trích xuất danh mục từ OneHotEncoder thành công.")
                else:
                    st.error(f"Lỗi: Số lượng danh mục ({len(onehot_encoder.categories_)}) không khớp số cột phân loại ({len(original_cat_features)}).")
                    all_loaded_successfully = False
            else:
                 st.error("Lỗi: Không tìm thấy thuộc tính 'categories_' trong OneHotEncoder.")
                 all_loaded_successfully = False
        except Exception as e_extract:
            st.error(f"Lỗi khi trích xuất thông tin từ preprocessor: {e_extract}")
            all_loaded_successfully = False
    except Exception as e_load_prep:
        st.error(f"Lỗi nghiêm trọng khi tải preprocessor: {e_load_prep}")
        return None, None, None, None, None

    # --- Tải Feature Names (sau khi xử lý) ---
    if not os.path.exists(features_path):
        st.error(f"LỖI: Không tìm thấy file tên đặc trưng tại '{features_path}'")
        all_loaded_successfully = False
    else:
        try:
            feature_names = joblib.load(features_path)
            print("Tải feature names (đã xử lý) thành công.")
            if isinstance(feature_names, np.ndarray): feature_names = feature_names.tolist()
            if not isinstance(feature_names, list):
                 st.warning(f"Định dạng feature_names không phải list (kiểu: {type(feature_names)}).")
                 try: feature_names = list(feature_names)
                 except TypeError:
                      st.error("Không thể chuyển đổi feature_names thành list.")
                      all_loaded_successfully = False
        except Exception as e_load_feat:
            st.error(f"Lỗi khi tải feature names: {e_load_feat}")
            all_loaded_successfully = False

    # --- Tải các Mô hình ML ---
    models_actually_loaded = 0
    for name, path in model_paths_dict.items():
        if not os.path.exists(path):
            st.warning(f"Cảnh báo: Không tìm thấy file mô hình '{name}' tại '{path}'. Bỏ qua.")
            continue
        try:
            model = joblib.load(path)
            loaded_models[name] = model
            models_actually_loaded += 1
            print(f"Tải mô hình ML {name} thành công.")
        except Exception as e_load_model:
            st.warning(f"Lỗi khi tải mô hình {name}: {e_load_model}. Bỏ qua.")

    if models_actually_loaded == 0:
         st.error("LỖI: Không tải được bất kỳ mô hình Machine Learning nào.")
         # Không nhất thiết phải dừng hoàn toàn nếu người dùng vẫn muốn dùng mô hình truyền thống
         # all_loaded_successfully = False # Tạm thời comment nếu muốn chạy tiếp chỉ với mô hình truyền thống

    if all_loaded_successfully and preprocessor and feature_names and original_cols_order and categorical_features_options:
        return preprocessor, feature_names, loaded_models, original_cols_order, categorical_features_options
    else:
        st.error("Một hoặc nhiều thành phần ML quan trọng không thể tải hoặc xử lý.")
        # Trả về những gì đã tải được để có thể vẫn dùng được phần khác nếu muốn
        return preprocessor, feature_names, loaded_models, original_cols_order, categorical_features_options

# --- Thực hiện tải ---
preprocessor, feature_names_out, ml_models, original_cols_order, categorical_features_options = load_all_artifacts(
    PREPROCESSOR_PATH, FEATURES_PATH, MODEL_PATHS
)

# --- Hàm tính toán cho mô hình truyền thống ---

def calculate_cocomo_basic(loc, mode, eaf, hrs_per_month):
    """Tính toán effort theo COCOMO II Basic (quy đổi ra person-hours)."""
    if loc <= 0:
        return "Lỗi (LOC <= 0)"
    if hrs_per_month <= 0:
        return "Lỗi (Giờ/Tháng <= 0)"

    kloc = loc / 1000.0

    # Tham số COCOMO II Basic
    params = {
        "Organic": {"a": 2.4, "b": 1.05},
        "Semi-detached": {"a": 3.0, "b": 1.12},
        "Embedded": {"a": 3.6, "b": 1.20}
    }

    if mode not in params:
        return "Lỗi (Chế độ không hợp lệ)"

    a = params[mode]["a"]
    b = params[mode]["b"]

    try:
        # Công thức cơ bản: Effort (Person-Months) = a * (KLOC)^b * EAF
        person_months = a * (kloc ** b) * eaf
        person_hours = person_months * hrs_per_month
        return max(0.0, round(person_hours, 2)) # Đảm bảo không âm
    except Exception as e:
        return f"Lỗi tính toán: {e}"

def calculate_fp_effort(fp, hrs_per_fp):
    """Tính toán effort dựa trên Function Points."""
    if fp <= 0:
        return "Lỗi (FP <= 0)"
    if hrs_per_fp <= 0:
        return "Lỗi (Giờ/FP <= 0)"
    try:
        person_hours = fp * hrs_per_fp
        return max(0.0, round(person_hours, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"

def calculate_ucp_effort(ucp, hrs_per_ucp):
    """Tính toán effort dựa trên Use Case Points."""
    if ucp <= 0:
        return "Lỗi (UCP <= 0)"
    if hrs_per_ucp <= 0:
        return "Lỗi (Giờ/UCP <= 0)"
    try:
        person_hours = ucp * hrs_per_ucp
        return max(0.0, round(person_hours, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"


# --- Tạo Giao diện Nhập liệu ---
# (Code giữ nguyên như trước)
st.sidebar.header("Nhập Thông tin Dự án")
input_values = {} # Dictionary chung để lưu tất cả giá trị người dùng nhập

# --- Widget nhập liệu cho ML và Mô hình truyền thống ---
# Nhóm các input cần cho cả ML và truyền thống lại với nhau
st.sidebar.subheader("Đặc trưng Cơ bản (Sử dụng bởi nhiều mô hình)")
col1, col2 = st.sidebar.columns(2)
with col1:
    # LOC cần cho ML (nếu có) và COCOMO
    if 'LOC' in original_cols_order or True: # Luôn hiển thị LOC vì cần cho COCOMO
        input_values['LOC'] = st.number_input("Lines of Code (LOC)", min_value=0, value=10000, step=100, key="loc_input")
    # FP cần cho ML (nếu có) và FP Estimation
    if 'FP' in original_cols_order or True: # Luôn hiển thị FP
        input_values['FP'] = st.number_input("Function Points (FP)", min_value=0, value=100, step=10, key="fp_input")
with col2:
     # UCP cần cho ML (nếu có) và UCP Estimation
    if 'UCP' in original_cols_order or True: # Luôn hiển thị UCP
        input_values['UCP'] = st.number_input("Use Case Points (UCP)", min_value=0.0, value=100.0, step=10.0, format="%.2f", key="ucp_input")

# --- Widget nhập liệu chỉ cho ML ---
# Chỉ hiển thị nếu preprocessor và các thành phần liên quan đã được tải
if preprocessor and original_cols_order and categorical_features_options:
    st.sidebar.subheader("Đặc trưng Bổ sung (Chủ yếu cho ML)")
    col_ml1, col_ml2 = st.sidebar.columns(2)
    with col_ml1:
        if 'Development Time (months)' in original_cols_order:
            input_values['Development Time (months)'] = st.number_input("Development Time (months)", min_value=1, value=6, step=1)
        # Thêm các input số khác cho ML nếu có
    with col_ml2:
        if 'Team Size' in original_cols_order:
            input_values['Team Size'] = st.number_input("Team Size", min_value=1, value=5, step=1)
        # Thêm các input số khác cho ML nếu có

    st.sidebar.subheader("Thông tin Phân loại (Chủ yếu cho ML)")
    col_cat1, col_cat2 = st.sidebar.columns(2)
    categorical_cols_with_options = list(categorical_features_options.keys())
    with col_cat1:
        for i, col_name in enumerate(categorical_cols_with_options):
            if col_name in original_cols_order and col_name in categorical_features_options:
                if i < len(categorical_cols_with_options) / 2:
                    options = categorical_features_options[col_name]
                    input_values[col_name] = st.selectbox(f"{col_name}", options=options, index=0, key=f"sb_{col_name}_1")
    with col_cat2:
        for i, col_name in enumerate(categorical_cols_with_options):
            if col_name in original_cols_order and col_name in categorical_features_options:
                if i >= len(categorical_cols_with_options) / 2:
                    options = categorical_features_options[col_name]
                    input_values[col_name] = st.selectbox(f"{col_name}", options=options, index=0, key=f"sb_{col_name}_2")
else:
    st.sidebar.warning("Không thể tải preprocessor hoặc thông tin cột ML. Phần nhập liệu cho ML bị vô hiệu hóa.")

# --- Widget nhập liệu cho Mô hình Truyền thống ---
st.sidebar.subheader("Tham số cho Mô hình Truyền thống")

# COCOMO II Basic
st.sidebar.markdown("**COCOMO II (Basic)**")
cocomo_mode = st.sidebar.selectbox("Chế độ Dự án", ["Organic", "Semi-detached", "Embedded"], key="cocomo_mode")
eaf = st.sidebar.number_input("Hệ số Điều chỉnh Nỗ lực (EAF)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="eaf", help="Effort Adjustment Factor - Tích các Cost Drivers. 1.0 là giá trị nominal.")
hours_per_month = st.sidebar.number_input("Số giờ làm việc/tháng (để quy đổi)", min_value=1, value=152, step=8, key="hrs_month", help="Dùng để chuyển đổi Person-Months từ COCOMO sang Person-Hours.")

# Function Points
st.sidebar.markdown("**Function Points (FP)**")
hours_per_fp = st.sidebar.number_input("Số giờ/Function Point (Năng suất)", min_value=0.1, value=10.0, step=0.5, format="%.1f", key="hrs_fp", help="Năng suất ước tính của nhóm phát triển (ví dụ: 5-20 giờ/FP tùy công nghệ, kinh nghiệm).")

# Use Case Points
st.sidebar.markdown("**Use Case Points (UCP)**")
hours_per_ucp = st.sidebar.number_input("Số giờ/Use Case Point (Năng suất)", min_value=0.1, value=20.0, step=1.0, format="%.1f", key="hrs_ucp", help="Hệ số năng suất cho UCP (thường trong khoảng 15-30 giờ/UCP).")


# --- Nút Dự đoán/Tính toán ---
calculate_button = st.sidebar.button("📊 Ước tính & So sánh Effort", use_container_width=True, type="primary")


# --- Xử lý và Hiển thị Kết quả ---
if calculate_button:
    st.divider()
    st.subheader("📊 Kết quả Ước tính Effort Tổng hợp")

    all_results = OrderedDict() # Lưu tất cả kết quả (ML và truyền thống)
    error_messages = {}       # Lưu lỗi của ML

    # --- 1. Dự đoán từ Mô hình Machine Learning ---
    # (Code giữ nguyên như trước)
    if preprocessor and feature_names_out and ml_models and original_cols_order and categorical_features_options:
        st.markdown("#### 1. Dự đoán từ Mô hình Machine Learning")
        # Tạo DataFrame cho ML input
        try:
            ordered_input_data_ml = {}
            missing_inputs_ml = []
            for col in original_cols_order: # Chỉ lấy các cột mà preprocessor cần
                 if col in input_values:
                      ordered_input_data_ml[col] = input_values[col]
                 else:
                      missing_inputs_ml.append(col)
                      ordered_input_data_ml[col] = np.nan # Giá trị thiếu sẽ được xử lý bởi imputer trong preprocessor

            if missing_inputs_ml:
                 st.warning(f"ML Input: Thiếu giá trị cho: {', '.join(missing_inputs_ml)}. Sẽ được xử lý bởi imputer.")

            input_df_ml = pd.DataFrame([ordered_input_data_ml], columns=original_cols_order)

            # Áp dụng preprocessor
            input_processed_np = preprocessor.transform(input_df_ml)

            # Chuyển thành DataFrame đã xử lý
            if isinstance(feature_names_out, list) and len(feature_names_out) == input_processed_np.shape[1]:
                 input_processed_df = pd.DataFrame(input_processed_np, columns=feature_names_out)

                 # Thực hiện dự đoán với các mô hình ML
                 for model_name, loaded_model in ml_models.items():
                     try:
                         pred = loaded_model.predict(input_processed_df)
                         prediction_value = float(pred[0]) if pred.size > 0 else 0.0
                         all_results[f"ML: {model_name}"] = max(0.0, round(prediction_value, 2)) # Thêm prefix "ML:"
                     except Exception as model_pred_e:
                         error_msg = f"Lỗi khi dự đoán bằng {model_name}: {str(model_pred_e)}"
                         st.error(error_msg)
                         all_results[f"ML: {model_name}"] = "Lỗi"
                         error_messages[model_name] = str(model_pred_e)

            else:
                 st.error(f"Lỗi ML: Số lượng tên đặc trưng ({len(feature_names_out)}) không khớp số cột sau transform ({input_processed_np.shape[1]}).")
                 for model_name in ml_models.keys(): # Đánh dấu lỗi cho tất cả ML models
                     all_results[f"ML: {model_name}"] = "Lỗi (Config)"

        except Exception as e_ml_process:
            st.error(f"Lỗi nghiêm trọng trong quá trình xử lý/dự đoán ML: {e_ml_process}")
            for model_name in ml_models.keys(): # Đánh dấu lỗi cho tất cả ML models
                 all_results[f"ML: {model_name}"] = "Lỗi (Process)"
            # import traceback # Đã import ở đầu
            print("--- TRACEBACK LỖI ML ---")
            print(traceback.format_exc())
            print("---------------------")
    else:
        st.info("Phần dự đoán Machine Learning không được thực hiện do thiếu thành phần cần thiết (preprocessor, models...).")


    # --- 2. Tính toán từ Mô hình Truyền thống ---
    # (Code giữ nguyên như trước)
    st.markdown("#### 2. Tính toán từ Mô hình Truyền thống")
    traditional_captions = [] # Lưu chú thích cho từng mô hình

    # Lấy giá trị input cần thiết
    loc_val = input_values.get('LOC', 0)
    fp_val = input_values.get('FP', 0)
    ucp_val = input_values.get('UCP', 0)

    # COCOMO II Basic
    cocomo_effort = calculate_cocomo_basic(loc_val, cocomo_mode, eaf, hours_per_month)
    all_results['COCOMO II (Basic)'] = cocomo_effort
    traditional_captions.append(f"* **COCOMO II (Basic):** Mode={cocomo_mode}, LOC={loc_val}, EAF={eaf}, Hours/Month={hours_per_month}")

    # Function Points
    fp_effort = calculate_fp_effort(fp_val, hours_per_fp)
    all_results['Function Points'] = fp_effort
    traditional_captions.append(f"* **Function Points:** FP={fp_val}, Hours/FP={hours_per_fp}")

    # Use Case Points
    ucp_effort = calculate_ucp_effort(ucp_val, hours_per_ucp)
    all_results['Use Case Points'] = ucp_effort
    traditional_captions.append(f"* **Use Case Points:** UCP={ucp_val}, Hours/UCP={hours_per_ucp}")

    st.markdown("**Tham số sử dụng:**")
    for caption in traditional_captions:
        st.markdown(caption)
    st.caption("Lưu ý: Kết quả 'Lỗi' xuất hiện nếu đầu vào không hợp lệ (ví dụ: LOC, FP, UCP <= 0 hoặc năng suất <= 0).")


    # --- 3. Hiển thị Bảng và Biểu đồ So sánh Tổng hợp ---
    st.markdown("#### 3. Bảng và Biểu đồ So sánh")

    if all_results:
        # Chuyển dictionary thành DataFrame
        result_list = []
        for model_name, effort in all_results.items():
             result_list.append({'Mô Hình Ước Tính': model_name, 'Effort Dự đoán (person-hours)': effort})
        result_df = pd.DataFrame(result_list)

        # Định dạng cột số và xử lý giá trị "Lỗi" cho bảng
        def format_effort_display(x):
            if isinstance(x, (int, float)):
                return f"{x:,.2f}" # Định dạng số
            return str(x) # Giữ nguyên chuỗi (ví dụ: "Lỗi...")

        st.write("Bảng so sánh kết quả:")
        st.dataframe(
             result_df.style.format({'Effort Dự đoán (person-hours)': format_effort_display}),
             use_container_width=True,
             hide_index=True # Ẩn index của DataFrame
        )

        # *** SỬA LỖI BIỂU ĐỒ ***
        # Hiển thị biểu đồ cột so sánh
        st.write("Biểu đồ so sánh:")
        try:
             # Tạo bản sao để xử lý cho biểu đồ
             chart_df = result_df.copy()

             # 1. Chuyển đổi cột Effort sang số, lỗi thành NaN
             #    Đảm bảo cột là kiểu string trước khi thay thế để tránh lỗi Attribute
             chart_df['Effort Dự đoán (person-hours)'] = chart_df['Effort Dự đoán (person-hours)'].astype(str).str.replace(',', '', regex=False) # Xóa dấu phẩy nếu có
             chart_df['Effort Dự đoán (person-hours)'] = pd.to_numeric(chart_df['Effort Dự đoán (person-hours)'], errors='coerce')

             # 2. Lọc bỏ các hàng có giá trị NaN (lỗi hoặc không thể chuyển đổi)
             chart_df.dropna(subset=['Effort Dự đoán (person-hours)'], inplace=True)

             # 3. Kiểm tra nếu còn dữ liệu hợp lệ
             if not chart_df.empty:
                  # Đặt tên mô hình làm index cho biểu đồ
                  chart_data = chart_df.set_index('Mô Hình Ước Tính')['Effort Dự đoán (person-hours)']
                  st.bar_chart(chart_data)
             else:
                  st.info("Không có dự đoán/tính toán hợp lệ (kiểu số) để vẽ biểu đồ.")

        except Exception as chart_e:
             st.warning(f"Không thể vẽ biểu đồ so sánh: {chart_e}")
             print("--- TRACEBACK LỖI BIỂU ĐỒ ---")
             print(traceback.format_exc())
             print("--------------------------")
        # *** KẾT THÚC SỬA LỖI BIỂU ĐỒ ***

    else:
         st.warning("Không có kết quả nào để hiển thị.")

    # Hiển thị chi tiết lỗi ML nếu có
    if error_messages:
         st.subheader("⚠️ Chi tiết lỗi dự đoán ML:")
         for model_name, msg in error_messages.items():
              st.caption(f"**{model_name}:** {msg}")

    st.info("""
    **Lưu ý quan trọng:**
    * Kết quả từ các mô hình (ML và truyền thống) chỉ là **ước tính**. Effort thực tế có thể khác biệt đáng kể do nhiều yếu tố không được mô hình hóa.
    * Độ chính xác của mô hình ML phụ thuộc vào chất lượng dữ liệu huấn luyện.
    * Độ chính xác của mô hình truyền thống phụ thuộc vào việc chọn đúng tham số (EAF, chế độ COCOMO, năng suất FP/UCP) phù hợp với dự án và môi trường cụ thể.
    * Hãy sử dụng các kết quả này như một điểm tham khảo và kết hợp với kinh nghiệm thực tế để đưa ra quyết định cuối cùng.
    """)


# --- Xử lý trường hợp không tải được artifacts ban đầu ---
# (Giữ nguyên phần này)
elif not ml_models and not preprocessor: # Nếu cả ML models và preprocessor đều lỗi
     st.error("Không thể tải các thành phần cần thiết cho dự đoán Machine Learning (preprocessor, models).")
     st.info("Bạn vẫn có thể sử dụng các tính toán từ mô hình truyền thống nếu nhập đủ thông tin.")
elif not ml_models:
     st.warning("Không tải được bất kỳ mô hình Machine Learning nào. Chỉ có thể sử dụng tính toán từ mô hình truyền thống.")
     st.info(f"Kiểm tra các file .joblib trong thư mục '{OUTPUT_DIR}'.")
elif not preprocessor or not feature_names_out or not original_cols_order or not categorical_features_options:
     st.error("Không thể tải hoặc xử lý preprocessor hoặc thông tin đặc trưng cho ML.")
     st.info("Phần dự đoán Machine Learning sẽ không hoạt động. Bạn vẫn có thể sử dụng các tính toán từ mô hình truyền thống.")
     st.info(f"Kiểm tra các file '{PREPROCESSOR_PATH}' và '{FEATURES_PATH}'.")

# --- Chân trang ---
st.markdown("---")
st.caption("Ứng dụng demo được xây dựng với Streamlit, Scikit-learn, XGBoost và các mô hình ước tính truyền thống.")
