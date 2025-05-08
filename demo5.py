# -*- coding: utf-8 -*-
"""
app.py: Ứng dụng Web Streamlit để dự đoán Effort
(Bao gồm Mô hình ML, COCOMO II Basic, FP, UCP và So sánh).
Tích hợp chuyển đổi LOC/FP/UCP và tính toán Thời gian/Quy mô dự án.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import OrderedDict
import math # Cần cho COCOMO
import traceback # Thêm để in lỗi chi tiết

# Import các lớp cần thiết
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


# --- Cấu hình Trang và Tải Artifacts ---

st.set_page_config(page_title="So sánh Ước tính Effort Phần mềm", layout="wide")

st.title("Ứng dụng So sánh Ước tính Effort, Thời gian & Quy mô Dự án 📊")
st.write("""
Nhập thông tin dự án hoặc một chỉ số kích thước chính (LOC, FP, UCP) để nhận ước tính effort,
thời gian phát triển, quy mô đội ngũ từ nhiều mô hình Machine Learning
và các phương pháp truyền thống.
""")

# Định nghĩa đường dẫn
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
    loaded_models = OrderedDict()
    preprocessor = None
    feature_names = None
    categorical_features_options = {}
    original_cols_order = []
    # all_loaded_successfully = True # Không cần nữa, sẽ kiểm tra từng phần

    if not os.path.exists(preprocessor_path):
        st.error(f"LỖI: Không tìm thấy file preprocessor tại '{preprocessor_path}'")
        # return None, None, None, None, None # Giữ lại để có thể chạy mô hình truyền thống
    else:
        try:
            preprocessor = joblib.load(preprocessor_path)
            print("Tải preprocessor thành công.")
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
                        preprocessor = None # Vô hiệu hóa ML nếu có lỗi này
                else:
                    st.error("Lỗi: Không tìm thấy thuộc tính 'categories_' trong OneHotEncoder.")
                    preprocessor = None # Vô hiệu hóa ML
            except Exception as e_extract:
                st.error(f"Lỗi khi trích xuất thông tin từ preprocessor: {e_extract}")
                preprocessor = None # Vô hiệu hóa ML
        except Exception as e_load_prep:
            st.error(f"Lỗi nghiêm trọng khi tải preprocessor: {e_load_prep}")
            preprocessor = None # Đảm bảo preprocessor là None nếu lỗi


    if not os.path.exists(features_path) and preprocessor: # Chỉ cần features_path nếu có preprocessor
        st.error(f"LỖI: Không tìm thấy file tên đặc trưng tại '{features_path}' (cần cho preprocessor đã tải).")
        preprocessor = None # Vô hiệu hóa ML nếu thiếu features
    elif preprocessor: # Chỉ tải features_path nếu preprocessor được tải
        try:
            feature_names = joblib.load(features_path)
            print("Tải feature names (đã xử lý) thành công.")
            if isinstance(feature_names, np.ndarray): feature_names = feature_names.tolist()
            if not isinstance(feature_names, list):
                st.warning(f"Định dạng feature_names không phải list (kiểu: {type(feature_names)}).")
                try: feature_names = list(feature_names)
                except TypeError:
                    st.error("Không thể chuyển đổi feature_names thành list.")
                    preprocessor = None # Vô hiệu hóa ML
        except Exception as e_load_feat:
            st.error(f"Lỗi khi tải feature names: {e_load_feat}")
            preprocessor = None # Vô hiệu hóa ML


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

    if models_actually_loaded == 0 and preprocessor: # Chỉ báo lỗi nếu preprocessor mong muốn có model
         st.error("LỖI: Không tải được bất kỳ mô hình Machine Learning nào (nhưng preprocessor đã được tải).")
         # loaded_models = OrderedDict() # Không cần thiết phải xóa, có thể để trống

    # Nếu preprocessor không tải được, các thông tin liên quan đến ML cũng không dùng được
    if not preprocessor:
        feature_names = None
        original_cols_order = []
        categorical_features_options = {}
        # loaded_models = OrderedDict() # Không nhất thiết phải xóa models nếu preprocessor lỗi, chỉ là không dùng được

    return preprocessor, feature_names, loaded_models, original_cols_order, categorical_features_options


preprocessor, feature_names_out, ml_models, original_cols_order, categorical_features_options = load_all_artifacts(
    PREPROCESSOR_PATH, FEATURES_PATH, MODEL_PATHS
)

# --- Hàm tính toán cho mô hình truyền thống và chuyển đổi ---

def calculate_cocomo_basic(loc, mode, eaf, hrs_per_month_param):
    if not isinstance(loc, (int, float)) or loc <= 0: return "Lỗi (LOC <= 0 hoặc không hợp lệ)", None
    if hrs_per_month_param <= 0: return "Lỗi (Giờ/Tháng <= 0)", None
    kloc = loc / 1000.0
    params = {"Organic": {"a": 2.4, "b": 1.05, "c": 2.5, "d": 0.38},
              "Semi-detached": {"a": 3.0, "b": 1.12, "c": 2.5, "d": 0.35},
              "Embedded": {"a": 3.6, "b": 1.20, "c": 2.5, "d": 0.32}}
    if mode not in params: return "Lỗi (Chế độ không hợp lệ)", None
    a, b = params[mode]["a"], params[mode]["b"]
    try:
        person_months = a * (kloc ** b) * eaf
        person_hours = person_months * hrs_per_month_param
        return max(0.0, round(person_hours, 2)), max(0.0, round(person_months, 2))
    except Exception as e: return f"Lỗi tính toán COCOMO: {e}", None

def calculate_fp_effort(fp, hrs_per_fp_param):
    if not isinstance(fp, (int, float)) or fp <= 0: return "Lỗi (FP <= 0 hoặc không hợp lệ)"
    if hrs_per_fp_param <= 0: return "Lỗi (Giờ/FP <= 0)"
    try:
        person_hours = fp * hrs_per_fp_param
        return max(0.0, round(person_hours, 2))
    except Exception as e: return f"Lỗi tính toán FP: {e}"

def calculate_ucp_effort(ucp, hrs_per_ucp_param):
    if not isinstance(ucp, (int, float)) or ucp <= 0: return "Lỗi (UCP <= 0 hoặc không hợp lệ)"
    if hrs_per_ucp_param <= 0: return "Lỗi (Giờ/UCP <= 0)"
    try:
        person_hours = ucp * hrs_per_ucp_param
        return max(0.0, round(person_hours, 2))
    except Exception as e: return f"Lỗi tính toán UCP: {e}"

# --- Hàm chuyển đổi ---
def convert_loc_to_fp(loc_val, loc_per_fp_ratio):
    if loc_per_fp_ratio <= 0: return "Lỗi (LOC/FP ratio <=0)"
    return round(loc_val / loc_per_fp_ratio, 2) if isinstance(loc_val, (int,float)) and loc_val > 0 else 0

def convert_fp_to_loc(fp_val, loc_per_fp_ratio):
    if loc_per_fp_ratio <= 0: return "Lỗi (LOC/FP ratio <=0)"
    return round(fp_val * loc_per_fp_ratio, 0) if isinstance(fp_val, (int,float)) and fp_val > 0 else 0

def convert_ucp_to_fp(ucp_val, ucp_fp_factor_val):
    if ucp_fp_factor_val <= 0: return "Lỗi (UCP to FP factor <=0)"
    return round(ucp_val * ucp_fp_factor_val, 2) if isinstance(ucp_val, (int,float)) and ucp_val > 0 else 0

def convert_fp_to_ucp(fp_val, ucp_fp_factor_val):
    if ucp_fp_factor_val <= 0: return "Lỗi (UCP to FP factor <=0)"
    return round(fp_val / ucp_fp_factor_val, 2) if isinstance(fp_val, (int,float)) and fp_val > 0 else 0


# --- Hàm tính Thời gian Phát triển và Quy mô Đội ngũ ---
def calculate_development_time(effort_person_months, team_size_val, scheduling_factor=1.0):
    if not isinstance(effort_person_months, (int,float)) or effort_person_months <= 0 or team_size_val <= 0 : return "N/A"
    try:
        return round((effort_person_months / team_size_val) * scheduling_factor, 2)
    except: return "Lỗi"

def calculate_team_size(effort_person_months, dev_time_months_val, scheduling_factor=1.0):
    if not isinstance(effort_person_months, (int,float)) or effort_person_months <= 0 or dev_time_months_val <= 0: return "N/A"
    try:
        return round((effort_person_months / dev_time_months_val) * scheduling_factor, 2)
    except: return "Lỗi"

def calculate_cocomo_tdev(effort_person_months, mode):
    if not isinstance(effort_person_months, (int,float)) or effort_person_months <=0: return "N/A"
    params = {"Organic": {"c": 2.5, "d": 0.38},
              "Semi-detached": {"c": 2.5, "d": 0.35},
              "Embedded": {"c": 2.5, "d": 0.32}}
    if mode not in params: return "Lỗi mode"
    c, d = params[mode]["c"], params[mode]["d"]
    try:
        return round(c * (effort_person_months ** d), 2)
    except: return "Lỗi tính TDEV"


# --- Giao diện Nhập liệu ---
st.sidebar.header("📝 Nhập Thông tin Dự án")
input_values_sidebar = {} # Đổi tên để tránh nhầm lẫn với input_values ở global scope nếu có

# 1. Chọn metric chính và nhập giá trị
st.sidebar.subheader("1. Chỉ số Kích thước Chính & Chuyển đổi")
primary_metric_type = st.sidebar.selectbox(
    "Chọn chỉ số kích thước bạn muốn nhập:",
    ("LOC", "Function Points (FP)", "Use Case Points (UCP)"),
    key="primary_metric_type"
)

available_languages = {
    "Assembly": 320, "C": 128, "COBOL": 100, "Fortran": 100,
    "Pascal": 90, "Ada": 70, "C++": 55, "Java": 53, "C#": 54, "JavaScript": 47,
    "Python": 30, "Perl": 25, "SQL": 12, "HTML": 40,
    "PowerBuilder": 15, "Visual Basic": 35, "Khác (Tùy chỉnh)": 50
}
selected_language = st.sidebar.selectbox(
    "Ngôn ngữ lập trình chính (ảnh hưởng LOC/FP):",
    list(available_languages.keys()),
    index=list(available_languages.keys()).index("Java"),
    key="language_select"
)

if selected_language == "Khác (Tùy chỉnh)":
    loc_per_fp_ratio = st.sidebar.number_input("Tỷ lệ LOC / FP tùy chỉnh:", min_value=1, value=50, step=1, key="custom_loc_fp_ratio")
else:
    loc_per_fp_ratio = available_languages[selected_language]
    st.sidebar.caption(f"Tỷ lệ LOC/FP cho {selected_language}: {loc_per_fp_ratio}")
input_values_sidebar['loc_per_fp_ratio_used'] = loc_per_fp_ratio

ucp_to_fp_factor = st.sidebar.number_input("Hệ số UCP sang FP (ví dụ: 1 UCP = X FP):", min_value=0.1, value=2.5, step=0.1, format="%.2f", key="ucp_fp_factor", help="Giá trị tham khảo: 1.5 - 3.5")

if primary_metric_type == "LOC":
    input_values_sidebar['LOC_primary'] = st.sidebar.number_input("Lines of Code (LOC)", min_value=0, value=10000, step=100, key="loc_primary_input")
elif primary_metric_type == "Function Points (FP)":
    input_values_sidebar['FP_primary'] = st.sidebar.number_input("Function Points (FP)", min_value=0.0, value=100.0, step=10.0, format="%.2f", key="fp_primary_input")
else: # UCP
    input_values_sidebar['UCP_primary'] = st.sidebar.number_input("Use Case Points (UCP)", min_value=0.0, value=100.0, step=10.0, format="%.2f", key="ucp_primary_input")

st.sidebar.subheader("2. Tham số Mô hình Truyền thống")
cocomo_mode = st.sidebar.selectbox("Chế độ Dự án", ["Organic", "Semi-detached", "Embedded"], key="cocomo_mode")
eaf = st.sidebar.number_input("Hệ số Điều chỉnh Nỗ lực (EAF)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="eaf")
hours_per_month = st.sidebar.number_input("Số giờ làm việc/tháng (quy đổi)", min_value=1, value=152, step=8, key="hrs_month")
hours_per_fp = st.sidebar.number_input("Số giờ/Function Point (Năng suất FP)", min_value=0.1, value=10.0, step=0.5, format="%.1f", key="hrs_fp")
hours_per_ucp = st.sidebar.number_input("Số giờ/Use Case Point (Năng suất UCP)", min_value=0.1, value=20.0, step=1.0, format="%.1f", key="hrs_ucp")

# Store traditional model params in input_values_sidebar
input_values_sidebar['cocomo_mode'] = cocomo_mode
input_values_sidebar['eaf'] = eaf
input_values_sidebar['hours_per_month'] = hours_per_month
input_values_sidebar['hours_per_fp'] = hours_per_fp
input_values_sidebar['hours_per_ucp'] = hours_per_ucp


if preprocessor and original_cols_order:
    st.sidebar.subheader("3. Đặc trưng Bổ sung (cho ML)")
    ml_specific_inputs = {}
    col_ml1, col_ml2 = st.sidebar.columns(2)
    ml_numeric_cols = [col for col in original_cols_order if col not in ['LOC', 'FP', 'UCP'] and (col not in categorical_features_options)]
    ml_categorical_cols = [col for col in original_cols_order if col in categorical_features_options and col not in ['LOC', 'FP', 'UCP']]

    with col_ml1:
        for i, col_name in enumerate(ml_numeric_cols):
            if i < (len(ml_numeric_cols) + 1) / 2 :
                 default_val = 0.0
                 if "Time" in col_name or "Month" in col_name : default_val = 6.0
                 elif "Size" in col_name or "Team" in col_name: default_val = 5.0
                 ml_specific_inputs[col_name] = st.number_input(f"{col_name}", value=default_val, step=1.0, format="%.1f", key=f"ml_num_{col_name}")
    with col_ml2:
        for i, col_name in enumerate(ml_numeric_cols):
            if i >= (len(ml_numeric_cols) + 1) / 2 :
                 default_val = 0.0
                 if "Time" in col_name or "Month" in col_name : default_val = 6.0
                 elif "Size" in col_name or "Team" in col_name: default_val = 5.0
                 ml_specific_inputs[col_name] = st.number_input(f"{col_name}", value=default_val, step=1.0, format="%.1f", key=f"ml_num_{col_name}")

    if ml_categorical_cols:
        st.sidebar.markdown("**Thông tin Phân loại (cho ML)**")
        col_cat1_ml, col_cat2_ml = st.sidebar.columns(2) # Đổi tên biến để tránh trùng lặp
        with col_cat1_ml:
            for i, col_name in enumerate(ml_categorical_cols):
                if col_name in categorical_features_options:
                    if i < (len(ml_categorical_cols) +1) / 2:
                        options = categorical_features_options[col_name]
                        ml_specific_inputs[col_name] = st.selectbox(f"{col_name}", options=options, index=0, key=f"ml_cat_{col_name}_1")
        with col_cat2_ml:
            for i, col_name in enumerate(ml_categorical_cols):
                if col_name in categorical_features_options:
                    if i >= (len(ml_categorical_cols) + 1) / 2:
                        options = categorical_features_options[col_name]
                        ml_specific_inputs[col_name] = st.selectbox(f"{col_name}", options=options, index=0, key=f"ml_cat_{col_name}_2")
    input_values_sidebar.update(ml_specific_inputs)
else:
    st.sidebar.info("Phần nhập liệu cho ML bị vô hiệu hóa do preprocessor hoặc thông tin cột không được tải.")

st.sidebar.subheader("4. Tham số Thời gian & Quy mô")
desired_team_size = st.sidebar.number_input("Quy mô đội ngũ mong muốn (số người)", min_value=0.1, value=5.0, step=0.5, format="%.1f", key="desired_team_size")
desired_dev_time = st.sidebar.number_input("Thời gian phát triển mong muốn (tháng)", min_value=0.1, value=6.0, step=0.5, format="%.1f", key="desired_dev_time")
input_values_sidebar['desired_team_size'] = desired_team_size
input_values_sidebar['desired_dev_time'] = desired_dev_time

calculate_button = st.sidebar.button("📊 Ước tính & So sánh", use_container_width=True, type="primary")

# --- Xử lý và Hiển thị Kết quả ---
if calculate_button:
    st.divider()
    st.subheader("📊 Kết quả Ước tính Tổng hợp")

    actual_loc = None
    actual_fp = None
    actual_ucp = None
    conversion_details = []

    current_loc_per_fp = input_values_sidebar['loc_per_fp_ratio_used']
    current_ucp_to_fp_factor = ucp_to_fp_factor # Lấy từ input trực tiếp

    st.markdown("#### 0. Giá trị Kích thước Đầu vào & Chuyển đổi")
    if primary_metric_type == "LOC":
        actual_loc = input_values_sidebar.get('LOC_primary')
        if isinstance(actual_loc, (int, float)) and actual_loc > 0:
            actual_fp = convert_loc_to_fp(actual_loc, current_loc_per_fp)
            if isinstance(actual_fp, str): # Lỗi từ convert
                conversion_details.append(f"LOC ({actual_loc}) sang FP: {actual_fp}")
                actual_ucp = "Lỗi (do FP)"
            elif isinstance(actual_fp, (int,float)) and actual_fp > 0:
                actual_ucp = convert_fp_to_ucp(actual_fp, current_ucp_to_fp_factor)
                if isinstance(actual_ucp, str): conversion_details.append(f"FP ({actual_fp}) sang UCP: {actual_ucp}")
            else: actual_ucp = 0 # Hoặc "Không tính (FP=0)"
        else:
            actual_loc = actual_loc if actual_loc is not None else "Chưa nhập"
            actual_fp = "Không tính (LOC không hợp lệ)"
            actual_ucp = "Không tính (LOC không hợp lệ)"
        conversion_details.insert(0, f"**Đầu vào chính: LOC = {actual_loc}**")

    elif primary_metric_type == "Function Points (FP)":
        actual_fp = input_values_sidebar.get('FP_primary')
        if isinstance(actual_fp, (int, float)) and actual_fp > 0:
            actual_loc = convert_fp_to_loc(actual_fp, current_loc_per_fp)
            actual_ucp = convert_fp_to_ucp(actual_fp, current_ucp_to_fp_factor)
            if isinstance(actual_loc, str): conversion_details.append(f"FP ({actual_fp}) sang LOC: {actual_loc}")
            if isinstance(actual_ucp, str): conversion_details.append(f"FP ({actual_fp}) sang UCP: {actual_ucp}")
        else:
            actual_fp = actual_fp if actual_fp is not None else "Chưa nhập"
            actual_loc = "Không tính (FP không hợp lệ)"
            actual_ucp = "Không tính (FP không hợp lệ)"
        conversion_details.insert(0, f"**Đầu vào chính: FP = {actual_fp}**")

    else: # UCP
        actual_ucp = input_values_sidebar.get('UCP_primary')
        if isinstance(actual_ucp, (int, float)) and actual_ucp > 0:
            actual_fp = convert_ucp_to_fp(actual_ucp, current_ucp_to_fp_factor)
            if isinstance(actual_fp, str): # Lỗi
                conversion_details.append(f"UCP ({actual_ucp}) sang FP: {actual_fp}")
                actual_loc = "Lỗi (do FP)"
            elif isinstance(actual_fp, (int,float)) and actual_fp > 0:
                actual_loc = convert_fp_to_loc(actual_fp, current_loc_per_fp)
                if isinstance(actual_loc, str): conversion_details.append(f"FP ({actual_fp}) sang LOC: {actual_loc}")
            else: actual_loc = 0 # Hoặc "Không tính (FP=0)"
        else:
            actual_ucp = actual_ucp if actual_ucp is not None else "Chưa nhập"
            actual_fp = "Không tính (UCP không hợp lệ)"
            actual_loc = "Không tính (UCP không hợp lệ)"
        conversion_details.insert(0, f"**Đầu vào chính: UCP = {actual_ucp}**")

    conversion_details.append(f"Tỷ lệ LOC/FP sử dụng: {current_loc_per_fp} ({selected_language})")
    conversion_details.append(f"Hệ số UCP/FP sử dụng: {current_ucp_to_fp_factor}")

    col_conv1, col_conv2, col_conv3 = st.columns(3)
    with col_conv1: st.metric(label="Lines of Code (LOC)", value=f"{actual_loc:,.0f}" if isinstance(actual_loc, (int,float)) else str(actual_loc))
    with col_conv2: st.metric(label="Function Points (FP)", value=f"{actual_fp:,.2f}" if isinstance(actual_fp, (int,float)) else str(actual_fp))
    with col_conv3: st.metric(label="Use Case Points (UCP)", value=f"{actual_ucp:,.2f}" if isinstance(actual_ucp, (int,float)) else str(actual_ucp))
    for detail in conversion_details: st.caption(detail)

    all_results = OrderedDict()
    error_messages_ml = {}
    effort_person_months_map = {}

    if preprocessor and feature_names_out and ml_models and original_cols_order:
        st.markdown("#### 1. Dự đoán Effort từ Mô hình Machine Learning")
        try:
            ml_feature_inputs_for_df = {}
            for col in original_cols_order:
                if col == 'LOC': ml_feature_inputs_for_df[col] = actual_loc if isinstance(actual_loc, (int,float)) else np.nan
                elif col == 'FP': ml_feature_inputs_for_df[col] = actual_fp if isinstance(actual_fp, (int,float)) else np.nan
                elif col == 'UCP': ml_feature_inputs_for_df[col] = actual_ucp if isinstance(actual_ucp, (int,float)) else np.nan
                elif col in input_values_sidebar: ml_feature_inputs_for_df[col] = input_values_sidebar[col]
                else: ml_feature_inputs_for_df[col] = np.nan

            input_df_ml = pd.DataFrame([ml_feature_inputs_for_df], columns=original_cols_order)
            # st.caption("Dữ liệu đầu vào cho ML (trước pre-processing):"); st.dataframe(input_df_ml.astype(str))

            input_processed_np = preprocessor.transform(input_df_ml)
            if isinstance(feature_names_out, list) and len(feature_names_out) == input_processed_np.shape[1]:
                input_processed_df = pd.DataFrame(input_processed_np, columns=feature_names_out)
                for model_name, loaded_model in ml_models.items():
                    try:
                        pred = loaded_model.predict(input_processed_df)
                        pv = float(pred[0]) if pred.size > 0 else 0.0
                        all_results[f"ML: {model_name}"] = max(0.0, round(pv, 2))
                        if input_values_sidebar['hours_per_month'] > 0 and pv > 0:
                             effort_person_months_map[f"ML: {model_name}"] = round(pv / input_values_sidebar['hours_per_month'], 2)
                        else: effort_person_months_map[f"ML: {model_name}"] = "Lỗi (giờ/tháng)"
                    except Exception as model_pred_e:
                        all_results[f"ML: {model_name}"] = "Lỗi"; error_messages_ml[model_name] = str(model_pred_e)
                        effort_person_months_map[f"ML: {model_name}"] = "Lỗi"
            else:
                st.error(f"Lỗi ML: Feature names ({len(feature_names_out)}) không khớp cột sau transform ({input_processed_np.shape[1]}).")
                for model_name in ml_models.keys(): all_results[f"ML: {model_name}"] = "Lỗi (Config)"; effort_person_months_map[f"ML: {model_name}"] = "Lỗi (Config)"
        except Exception as e_ml_process:
            st.error(f"Lỗi xử lý/dự đoán ML: {e_ml_process}")
            for model_name in ml_models.keys(): all_results[f"ML: {model_name}"] = "Lỗi (Process)"; effort_person_months_map[f"ML: {model_name}"] = "Lỗi (Process)"
    else: st.info("Phần ML không thực hiện do thiếu thành phần.")

    st.markdown("#### 2. Ước tính Effort từ Mô hình Truyền thống")
    traditional_captions_display = []
    hpm = input_values_sidebar['hours_per_month'] # Viết tắt để dễ đọc

    cocomo_effort_hours, cocomo_effort_months = "N/A", "N/A"
    if isinstance(actual_loc, (int, float)) and actual_loc > 0:
        cocomo_effort_hours, cocomo_effort_months = calculate_cocomo_basic(actual_loc, input_values_sidebar['cocomo_mode'], input_values_sidebar['eaf'], hpm)
    all_results['COCOMO II (Basic)'] = cocomo_effort_hours
    effort_person_months_map['COCOMO II (Basic)'] = cocomo_effort_months
    traditional_captions_display.append(f"* **COCOMO II:** LOC={actual_loc}, Mode={input_values_sidebar['cocomo_mode']}, EAF={input_values_sidebar['eaf']}, Hrs/M={hpm}")

    fp_effort_hours = "N/A"
    if isinstance(actual_fp, (int, float)) and actual_fp > 0:
        fp_effort_hours = calculate_fp_effort(actual_fp, input_values_sidebar['hours_per_fp'])
    all_results['Function Points'] = fp_effort_hours
    if isinstance(fp_effort_hours, (int,float)) and hpm > 0 : effort_person_months_map['Function Points'] = round(fp_effort_hours / hpm, 2)
    else: effort_person_months_map['Function Points'] = "Lỗi" if not isinstance(fp_effort_hours, (int,float)) else "Lỗi (hpm)"
    traditional_captions_display.append(f"* **FP:** FP={actual_fp}, Hrs/FP={input_values_sidebar['hours_per_fp']}")

    ucp_effort_hours = "N/A"
    if isinstance(actual_ucp, (int, float)) and actual_ucp > 0:
        ucp_effort_hours = calculate_ucp_effort(actual_ucp, input_values_sidebar['hours_per_ucp'])
    all_results['Use Case Points'] = ucp_effort_hours
    if isinstance(ucp_effort_hours, (int,float)) and hpm > 0: effort_person_months_map['Use Case Points'] = round(ucp_effort_hours / hpm, 2)
    else: effort_person_months_map['Use Case Points'] = "Lỗi" if not isinstance(ucp_effort_hours, (int,float)) else "Lỗi (hpm)"
    traditional_captions_display.append(f"* **UCP:** UCP={actual_ucp}, Hrs/UCP={input_values_sidebar['hours_per_ucp']}")
    st.markdown("**Tham số sử dụng cho mô hình truyền thống:**"); [st.markdown(c) for c in traditional_captions_display]

    st.markdown("#### 3. Bảng và Biểu đồ So sánh Effort")

    # Sửa lỗi PyArrow bằng cách chuyển đổi cột sang string trước khi hiển thị
    def safe_format_display(x):
        if isinstance(x, (int, float)): return f"{x:,.2f}"
        return str(x) # Đảm bảo mọi thứ là string

    if all_results:
        result_list_effort = [{'Mô Hình Ước Tính': name,
                               'Effort (person-hours)': val,
                               'Effort (person-months)': effort_person_months_map.get(name, "N/A")}
                              for name, val in all_results.items()]
        result_df_effort = pd.DataFrame(result_list_effort)

        # Chuyển đổi các cột sang string để hiển thị an toàn với st.dataframe
        df_effort_display = result_df_effort.copy()
        df_effort_display['Effort (person-hours)'] = df_effort_display['Effort (person-hours)'].apply(safe_format_display)
        df_effort_display['Effort (person-months)'] = df_effort_display['Effort (person-months)'].apply(safe_format_display)

        st.write("Bảng so sánh Effort:"); st.dataframe(df_effort_display, use_container_width=True, hide_index=True)

        st.write("Biểu đồ so sánh Effort (person-hours):")
        try:
            chart_df_effort = result_df_effort.copy() # Dùng DataFrame gốc cho biểu đồ
            chart_df_effort['Effort (person-hours)'] = pd.to_numeric(chart_df_effort['Effort (person-hours)'], errors='coerce')
            chart_df_effort.dropna(subset=['Effort (person-hours)'], inplace=True)
            if not chart_df_effort.empty:
                st.bar_chart(chart_df_effort.set_index('Mô Hình Ước Tính')['Effort (person-hours)'])
            else: st.info("Không có dữ liệu effort hợp lệ để vẽ biểu đồ.")
        except Exception as chart_e: st.warning(f"Không thể vẽ biểu đồ effort: {chart_e}")
    else: st.warning("Không có kết quả effort nào.")

    st.markdown("#### 4. Ước tính Thời gian Phát triển & Quy mô Đội ngũ")
    if effort_person_months_map:
        time_size_results_list = []
        has_valid_effort_pm = any(isinstance(val, (int, float)) and val > 0 for val in effort_person_months_map.values())
        if not has_valid_effort_pm: st.warning("Không có effort (person-months) hợp lệ để tính Thời gian/Quy mô.")
        else:
            for model_name, effort_pm_val in effort_person_months_map.items():
                cocomo_tdev_val, dev_time_calc, team_size_calc = "N/A", "N/A", "N/A"
                if isinstance(effort_pm_val, (int,float)) and effort_pm_val > 0:
                    if model_name == 'COCOMO II (Basic)':
                        cocomo_tdev_val = calculate_cocomo_tdev(effort_pm_val, input_values_sidebar['cocomo_mode'])
                    if input_values_sidebar['desired_team_size'] > 0:
                        dev_time_calc = calculate_development_time(effort_pm_val, input_values_sidebar['desired_team_size'])
                    if input_values_sidebar['desired_dev_time'] > 0:
                        team_size_calc = calculate_team_size(effort_pm_val, input_values_sidebar['desired_dev_time'])
                time_size_results_list.append({
                    'Mô Hình Effort': model_name,
                    'Effort (person-months)': effort_pm_val,
                    'TDEV (tháng) - COCOMO': cocomo_tdev_val,
                    f'TDEV (tháng) với Team={input_values_sidebar["desired_team_size"]}': dev_time_calc,
                    f'Team Size cho TDEV={input_values_sidebar["desired_dev_time"]} tháng': team_size_calc
                })
            if time_size_results_list:
                result_df_time_size = pd.DataFrame(time_size_results_list)
                # Chuyển đổi tất cả các cột sang string để hiển thị an toàn
                df_time_size_display = result_df_time_size.astype(str) # Cách đơn giản nhất
                # Hoặc dùng safe_format_display cho từng cột nếu cần định dạng số cụ thể
                # for col in df_time_size_display.columns:
                #    df_time_size_display[col] = df_time_size_display[col].apply(safe_format_display)

                st.write("Bảng ước tính Thời gian & Quy mô:")
                st.dataframe(df_time_size_display, use_container_width=True, hide_index=True)
                st.caption(f"Giờ/tháng quy đổi: {hpm}. COCOMO TDEV chỉ áp dụng cho COCOMO II. Các tính toán khác giả định Effort = Team * Time.")
            else: st.info("Không có kết quả Thời gian/Quy mô.")
    else: st.warning("Không có dữ liệu effort (person-months) để tính Thời gian/Quy mô.")

    if error_messages_ml:
        st.subheader("⚠️ Chi tiết lỗi dự đoán ML:"); [st.caption(f"**{name}:** {msg}") for name, msg in error_messages_ml.items()]
    st.info("""**Lưu ý quan trọng:** Kết quả chỉ là **ước tính**. Độ chính xác phụ thuộc vào dữ liệu và tham số. Sử dụng như một điểm tham khảo.""")

elif not calculate_button:
    # Chỉ hiển thị thông báo lỗi tải artifact khi chưa nhấn nút và có vấn đề
    if not preprocessor and (os.path.exists(PREPROCESSOR_PATH) or os.path.exists(FEATURES_PATH)):
         st.warning("Không thể tải preprocessor hoặc feature names cho ML. Chức năng ML có thể bị hạn chế.")
    if not ml_models and any(os.path.exists(p) for p in MODEL_PATHS.values()):
         st.warning("Không thể tải một hoặc nhiều mô hình ML. Chức năng ML có thể bị hạn chế.")


st.markdown("---"); st.caption("Ứng dụng demo.")