# -*- coding: utf-8 -*-
"""
app.py: Ứng dụng Web Streamlit để dự đoán Effort
(Bao gồm Mô hình ML, COCOMO II Basic, FP, UCP, chuyển đổi LOC/UCP/FP, Development Time, Team Size và So sánh).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import OrderedDict
import math
import traceback
import uuid  # Để tạo UUID cho artifact_id nếu cần

# Import các lớp cần thiết
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
except ImportError as e:
    st.error(f"Lỗi Import thư viện: {e}. Hãy đảm bảo các thư viện cần thiết đã được cài đặt.")
    st.stop()

# --- Cấu hình Trang ---
st.set_page_config(page_title="So sánh Ước tính Effort Phần mềm", layout="wide")
st.title("Ứng dụng So sánh Ước tính Effort Phần mềm 📊")
st.write("""
Nhập thông tin dự án để nhận ước tính effort (person-hours), development time (tháng), team size,
và chuyển đổi giữa LOC, UCP, FP từ nhiều mô hình Machine Learning và phương pháp truyền thống.
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
    """
    Tải preprocessor, feature names, các mô hình ML, và trích xuất các danh mục.
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
                    st.error(f"Lỗi: Số lượng danh mục không khớp số cột phân loại.")
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

    # --- Tải Feature Names ---
    if not os.path.exists(features_path):
        st.error(f"LỖI: Không tìm thấy file tên đặc trưng tại '{features_path}'")
        all_loaded_successfully = False
    else:
        try:
            feature_names = joblib.load(features_path)
            print("Tải feature names thành công.")
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
            if not isinstance(feature_names, list):
                st.warning(f"Định dạng feature_names không phải list.")
                try:
                    feature_names = list(feature_names)
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
        all_loaded_successfully = False

    return preprocessor, feature_names, loaded_models, original_cols_order, categorical_features_options

# --- Hàm chuyển đổi giữa LOC, FP, UCP ---
def loc_to_fp(loc, loc_per_fp):
    """Chuyển đổi từ LOC sang FP."""
    if loc <= 0 or loc_per_fp <= 0:
        return "Lỗi (LOC hoặc hệ số <= 0)"
    try:
        fp = loc / loc_per_fp
        return max(0.0, round(fp, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"

def fp_to_loc(fp, loc_per_fp):
    """Chuyển đổi từ FP sang LOC."""
    if fp <= 0 or loc_per_fp <= 0:
        return "Lỗi (FP hoặc hệ số <= 0)"
    try:
        loc = fp * loc_per_fp
        return max(0.0, round(loc, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"

def fp_to_ucp(fp, fp_per_ucp):
    """Chuyển đổi từ FP sang UCP."""
    if fp <= 0 or fp_per_ucp <= 0:
        return "Lỗi (FP hoặc hệ số <= 0)"
    try:
        ucp = fp / fp_per_ucp
        return max(0.0, round(ucp, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"

def ucp_to_fp(ucp, fp_per_ucp):
    """Chuyển đổi từ UCP sang FP."""
    if ucp <= 0 or fp_per_ucp <= 0:
        return "Lỗi (UCP hoặc hệ số <= 0)"
    try:
        fp = ucp * fp_per_ucp
        return max(0.0, round(fp, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"

# --- Hàm tính toán cho mô hình truyền thống ---
def calculate_cocomo_basic(loc, mode, eaf, hrs_per_month):
    """Tính toán effort theo COCOMO II Basic (quy đổi ra person-hours)."""
    if loc <= 0:
        return "Lỗi (LOC <= 0)"
    if hrs_per_month <= 0:
        return "Lỗi (Giờ/Tháng <= 0)"
    kloc = loc / 1000.0
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
        person_months = a * (kloc ** b) * eaf
        person_hours = person_months * hrs_per_month
        return max(0.0, round(person_hours, 2))
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

def calculate_development_time(effort, team_size, hrs_per_month):
    """Tính development time (tháng) từ effort."""
    if effort <= 0 or team_size <= 0 or hrs_per_month <= 0:
        return "Lỗi (Effort, Team Size hoặc Giờ/Tháng <= 0)"
    try:
        time_months = effort / (team_size * hrs_per_month)
        return max(0.0, round(time_months, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"

def calculate_team_size(effort, dev_time, hrs_per_month):
    """Tính team size từ effort và development time."""
    if effort <= 0 or dev_time <= 0 or hrs_per_month <= 0:
        return "Lỗi (Effort, Dev Time hoặc Giờ/Tháng <= 0)"
    try:
        team_size = effort / (dev_time * hrs_per_month)
        return max(0.0, round(team_size, 2))
    except Exception as e:
        return f"Lỗi tính toán: {e}"

# --- Tải Artifacts ---
preprocessor, feature_names_out, ml_models, original_cols_order, categorical_features_options = load_all_artifacts(
    PREPROCESSOR_PATH, FEATURES_PATH, MODEL_PATHS
)

# --- Tạo Giao diện Nhập liệu ---
st.sidebar.header("Nhập Thông tin Dự án")
input_values = {}

# --- Widget nhập liệu cho ML và Mô hình truyền thống ---
st.sidebar.subheader("Đặc trưng Cơ bản (Sử dụng bởi nhiều mô hình)")
col1, col2 = st.sidebar.columns(2)
with col1:
    input_values['LOC'] = st.number_input("Lines of Code (LOC)", min_value=0, value=10000, step=100, key="loc_input")
    input_values['FP'] = st.number_input("Function Points (FP)", min_value=0, value=100, step=10, key="fp_input")
with col2:
    input_values['UCP'] = st.number_input("Use Case Points (UCP)", min_value=0.0, value=100.0, step=10.0, format="%.2f", key="ucp_input")

# --- Widget nhập liệu chỉ cho ML ---
if preprocessor and original_cols_order and categorical_features_options:
    st.sidebar.subheader("Đặc trưng Bổ sung (Chủ yếu cho ML)")
    col_ml1, col_ml2 = st.sidebar.columns(2)
    with col_ml1:
        if 'Development Time (months)' in original_cols_order:
            input_values['Development Time (months)'] = st.number_input("Development Time (months)", min_value=1, value=6, step=1)
    with col_ml2:
        if 'Team Size' in original_cols_order:
            input_values['Team Size'] = st.number_input("Team Size", min_value=1, value=5, step=1)

    st.sidebar.subheader("Thông tin Phân loại (Chủ yếu cho ML)")
    col_cat1, col_cat2 = st.sidebar.columns(2)
    categorical_cols_with_options = list(categorical_features_options.keys())
    with col_cat1:
        for i, col_name in enumerate(categorical_cols_with_options[:len(categorical_cols_with_options)//2]):
            if col_name in original_cols_order:
                options = categorical_features_options[col_name]
                input_values[col_name] = st.selectbox(f"{col_name}", options=options, index=0, key=f"sb_{col_name}_1")
    with col_cat2:
        for i, col_name in enumerate(categorical_cols_with_options[len(categorical_cols_with_options)//2:]):
            if col_name in original_cols_order:
                options = categorical_features_options[col_name]
                input_values[col_name] = st.selectbox(f"{col_name}", options=options, index=0, key=f"sb_{col_name}_2")
else:
    st.sidebar.warning("Không thể tải preprocessor hoặc thông tin cột ML. Phần nhập liệu cho ML bị vô hiệu hóa.")

# --- Widget nhập liệu cho Mô hình Truyền thống ---
st.sidebar.subheader("Tham số cho Mô hình Truyền thống")

# Chuyển đổi LOC, FP, UCP
st.sidebar.markdown("**Chuyển đổi LOC/FP/UCP**")
loc_per_fp = st.sidebar.number_input("LOC per FP (Java: ~53)", min_value=0.1, value=53.0, step=1.0, format="%.1f", key="loc_per_fp")
fp_per_ucp = st.sidebar.number_input("FP per UCP (~5-15)", min_value=0.1, value=10.0, step=1.0, format="%.1f", key="fp_per_ucp")

# COCOMO II Basic
st.sidebar.markdown("**COCOMO II (Basic)**")
cocomo_mode = st.sidebar.selectbox("Chế độ Dự án", ["Organic", "Semi-detached", "Embedded"], key="cocomo_mode")
eaf = st.sidebar.number_input("Hệ số Điều chỉnh Nỗ lực (EAF)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="eaf")
hours_per_month = st.sidebar.number_input("Số giờ làm việc/tháng", min_value=1, value=152, step=8, key="hrs_month")

# Function Points
st.sidebar.markdown("**Function Points (FP)**")
hours_per_fp = st.sidebar.number_input("Số giờ/Function Point", min_value=0.1, value=10.0, step=0.5, format="%.1f", key="hrs_fp")

# Use Case Points
st.sidebar.markdown("**Use Case Points (UCP)**")
hours_per_ucp = st.sidebar.number_input("Số giờ/Use Case Point", min_value=0.1, value=20.0, step=1.0, format="%.1f", key="hrs_ucp")

# Development Time và Team Size
st.sidebar.markdown("**Development Time & Team Size**")
team_size_input = st.sidebar.number_input("Team Size (dùng để tính Dev Time)", min_value=1, value=5, step=1, key="team_size_input")
dev_time_input = st.sidebar.number_input("Development Time (tháng, dùng để tính Team Size)", min_value=0.1, value=6.0, step=0.1, format="%.1f", key="dev_time_input")

# --- Nút Dự đoán/Tính toán ---
calculate_button = st.sidebar.button("📊 Ước tính & So sánh", use_container_width=True, type="primary")

# --- Xử lý và Hiển thị Kết quả ---
if calculate_button:
    st.divider()
    st.subheader("📊 Kết quả Ước tính Tổng hợp")
    all_results = OrderedDict()
    error_messages = {}

    # --- 1. Chuyển đổi LOC, FP, UCP ---
    st.markdown("#### 1. Kết quả Chuyển đổi LOC/FP/UCP")
    loc_val = input_values.get('LOC', 0)
    fp_val = input_values.get('FP', 0)
    ucp_val = input_values.get('UCP', 0)

    conversions = {
        "FP từ LOC": loc_to_fp(loc_val, loc_per_fp),
        "LOC từ FP": fp_to_loc(fp_val, loc_per_fp),
        "UCP từ FP": fp_to_ucp(fp_val, fp_per_ucp),
        "FP từ UCP": ucp_to_fp(ucp_val, fp_per_ucp)
    }
    conversion_df = pd.DataFrame(list(conversions.items()), columns=['Chuyển đổi', 'Kết quả'])
    st.dataframe(conversion_df, use_container_width=True, hide_index=True)
    st.caption(f"Tham số: LOC per FP = {loc_per_fp}, FP per UCP = {fp_per_ucp}")

    # --- 2. Dự đoán từ Mô hình Machine Learning ---
    if preprocessor and feature_names_out and ml_models and original_cols_order and categorical_features_options:
        st.markdown("#### 2. Dự đoán từ Mô hình Machine Learning")
        try:
            ordered_input_data_ml = {}
            missing_inputs_ml = []
            for col in original_cols_order:
                if col in input_values:
                    ordered_input_data_ml[col] = input_values[col]
                else:
                    missing_inputs_ml.append(col)
                    ordered_input_data_ml[col] = np.nan
            if missing_inputs_ml:
                st.warning(f"ML Input: Thiếu giá trị cho: {', '.join(missing_inputs_ml)}.")
            input_df_ml = pd.DataFrame([ordered_input_data_ml], columns=original_cols_order)
            input_processed_np = preprocessor.transform(input_df_ml)
            if isinstance(feature_names_out, list) and len(feature_names_out) == input_processed_np.shape[1]:
                input_processed_df = pd.DataFrame(input_processed_np, columns=feature_names_out)
                for model_name, loaded_model in ml_models.items():
                    try:
                        pred = loaded_model.predict(input_processed_df)
                        prediction_value = float(pred[0]) if pred.size > 0 else 0.0
                        all_results[f"ML: {model_name}"] = {
                            'Effort (person-hours)': max(0.0, round(prediction_value, 2)),
                            'Dev Time (months)': calculate_development_time(prediction_value, team_size_input, hours_per_month),
                            'Team Size': calculate_team_size(prediction_value, dev_time_input, hours_per_month)
                        }
                    except Exception as model_pred_e:
                        error_messages[model_name] = str(model_pred_e)
                        all_results[f"ML: {model_name}"] = {'Effort (person-hours)': "Lỗi", 'Dev Time (months)': "Lỗi", 'Team Size': "Lỗi"}
            else:
                st.error(f"Lỗi ML: Số lượng tên đặc trưng không khớp.")
                for model_name in ml_models.keys():
                    all_results[f"ML: {model_name}"] = {'Effort (person-hours)': "Lỗi (Config)", 'Dev Time (months)': "Lỗi", 'Team Size': "Lỗi"}
        except Exception as e_ml_process:
            st.error(f"Lỗi nghiêm trọng trong ML: {e_ml_process}")
            for model_name in ml_models.keys():
                all_results[f"ML: {model_name}"] = {'Effort (person-hours)': "Lỗi (Process)", 'Dev Time (months)': "Lỗi", 'Team Size': "Lỗi"}
            print(traceback.format_exc())
    else:
        st.info("Phần dự đoán ML không thực hiện do thiếu thành phần cần thiết.")

    # --- 3. Tính toán từ Mô hình Truyền thống ---
    st.markdown("#### 3. Tính toán từ Mô hình Truyền thống")
    traditional_captions = []

    # COCOMO II Basic
    cocomo_effort = calculate_cocomo_basic(loc_val, cocomo_mode, eaf, hours_per_month)
    cocomo_dev_time = calculate_development_time(cocomo_effort, team_size_input, hours_per_month) if isinstance(cocomo_effort, (int, float)) else "Lỗi"
    cocomo_team_size = calculate_team_size(cocomo_effort, dev_time_input, hours_per_month) if isinstance(cocomo_effort, (int, float)) else "Lỗi"
    all_results['COCOMO II (Basic)'] = {
        'Effort (person-hours)': cocomo_effort,
        'Dev Time (months)': cocomo_dev_time,
        'Team Size': cocomo_team_size
    }
    traditional_captions.append(f"* **COCOMO II (Basic):** Mode={cocomo_mode}, LOC={loc_val}, EAF={eaf}, Hours/Month={hours_per_month}")

    # Function Points
    fp_effort = calculate_fp_effort(fp_val, hours_per_fp)
    fp_dev_time = calculate_development_time(fp_effort, team_size_input, hours_per_month) if isinstance(fp_effort, (int, float)) else "Lỗi"
    fp_team_size = calculate_team_size(fp_effort, dev_time_input, hours_per_month) if isinstance(fp_effort, (int, float)) else "Lỗi"
    all_results['Function Points'] = {
        'Effort (person-hours)': fp_effort,
        'Dev Time (months)': fp_dev_time,
        'Team Size': fp_team_size
    }
    traditional_captions.append(f"* **Function Points:** FP={fp_val}, Hours/FP={hours_per_fp}")

    # Use Case Points
    ucp_effort = calculate_ucp_effort(ucp_val, hours_per_ucp)
    ucp_dev_time = calculate_development_time(ucp_effort, team_size_input, hours_per_month) if isinstance(ucp_effort, (int, float)) else "Lỗi"
    ucp_team_size = calculate_team_size(ucp_effort, dev_time_input, hours_per_month) if isinstance(ucp_effort, (int, float)) else "Lỗi"
    all_results['Use Case Points'] = {
        'Effort (person-hours)': ucp_effort,
        'Dev Time (months)': ucp_dev_time,
        'Team Size': ucp_team_size
    }
    traditional_captions.append(f"* **Use Case Points:** UCP={ucp_val}, Hours/UCP={hours_per_ucp}")

    st.markdown("**Tham số sử dụng:**")
    for caption in traditional_captions:
        st.markdown(caption)
    st.caption("Lưu ý: Kết quả 'Lỗi' xuất hiện nếu đầu vào không hợp lệ.")

    # --- 4. Hiển thị Bảng và Biểu đồ So sánh ---
    st.markdown("#### 4. Bảng và Biểu đồ So sánh")
    if all_results:
        result_list = []
        for model_name, metrics in all_results.items():
            result_list.append({
                'Mô Hình Ước Tính': model_name,
                'Effort (person-hours)': metrics['Effort (person-hours)'],
                'Dev Time (months)': metrics['Dev Time (months)'],
                'Team Size': metrics['Team Size']
            })
        result_df = pd.DataFrame(result_list)

        def format_display(x):
            if isinstance(x, (int, float)):
                return f"{x:,.2f}"
            return str(x)
        st.write("Bảng so sánh kết quả:")
        st.dataframe(
            result_df.style.format({
                'Effort (person-hours)': format_display,
                'Dev Time (months)': format_display,
                'Team Size': format_display
            }),
            use_container_width=True,
            hide_index=True
        )

        st.write("Biểu đồ so sánh Effort:")
        try:
            chart_df = result_df.copy()
            chart_df['Effort (person-hours)'] = pd.to_numeric(chart_df['Effort (person-hours)'].astype(str).str.replace(',', '', regex=False), errors='coerce')
            chart_df.dropna(subset=['Effort (person-hours)'], inplace=True)
            if not chart_df.empty:
                chart_data = chart_df.set_index('Mô Hình Ước Tính')['Effort (person-hours)']
                st.bar_chart(chart_data)
            else:
                st.info("Không có dự đoán/tính toán hợp lệ để vẽ biểu đồ Effort.")
        except Exception as chart_e:
            st.warning(f"Không thể vẽ biểu đồ Effort: {chart_e}")
            print(traceback.format_exc())

        st.write("Biểu đồ so sánh Development Time:")
        try:
            chart_df = result_df.copy()
            chart_df['Dev Time (months)'] = pd.to_numeric(chart_df['Dev Time (months)'].astype(str).str.replace(',', '', regex=False), errors='coerce')
            chart_df.dropna(subset=['Dev Time (months)'], inplace=True)
            if not chart_df.empty:
                chart_data = chart_df.set_index('Mô Hình Ước Tính')['Dev Time (months)']
                st.bar_chart(chart_data)
            else:
                st.info("Không có dự đoán/tính toán hợp lệ để vẽ biểu đồ Dev Time.")
        except Exception as chart_e:
            st.warning(f"Không thể vẽ biểu đồ Dev ^^Time: {chart_e}")
            print(traceback.format_exc())

        st.write("Biểu đồ so sánh Team Size:")
        try:
            chart_df = result_df.copy()
            chart_df['Team Size'] = pd.to_numeric(chart_df['Team Size'].astype(str).str.replace(',', '', regex=False), errors='coerce')
            chart_df.dropna(subset=['Team Size'], inplace=True)
            if not chart_df.empty:
                chart_data = chart_df.set_index('Mô Hình Ước Tính')['Team Size']
                st.bar_chart(chart_data)
            else:
                st.info("Không có dự đoán/tính toán hợp lệ để vẽ biểu đồ Team Size.")
        except Exception as chart_e:
            st.warning(f"Không thể vẽ biểu đồ Team Size: {chart_e}")
            print(traceback.format_exc())

    else:
        st.warning("Không có kết quả nào để hiển thị.")

    if error_messages:
        st.subheader("⚠️ Chi tiết lỗi dự đoán ML:")
        for model_name, msg in error_messages.items():
            st.caption(f"**{model_name}:** {msg}")

    st.info("""
    **Lưu ý quan trọng:**
    * Kết quả chỉ là **ước tính**. Effort, thời gian và đội ngũ thực tế có thể khác biệt.
    * Độ chính xác của ML phụ thuộc vào dữ liệu huấn luyện.
    * Độ chính xác của mô hình truyền thống phụ thuộc vào tham số (EAF, năng suất FP/UCP, hệ số chuyển đổi).
    * Sử dụng kết quả như tham khảo và kết hợp với kinh nghiệm thực tế.
    """)

# --- Xử lý trường hợp không tải được artifacts ---
elif not ml_models and not preprocessor:
    st.error("Không thể tải các thành phần cần thiết cho dự đoán ML.")
    st.info("Bạn vẫn có thể sử dụng các tính toán truyền thống nếu nhập đủ thông tin.")
elif not ml_models:
    st.warning("Không tải được mô hình ML. Chỉ sử dụng tính toán truyền thống.")
    st.info(f"Kiểm tra các file .joblib trong thư mục '{OUTPUT_DIR}'.")
elif not preprocessor or not feature_names_out or not original_cols_order or not categorical_features_options:
    st.error("Không thể tải preprocessor hoặc thông tin đặc trưng cho ML.")
    st.info("Phần dự đoán ML không hoạt động. Vẫn có thể sử dụng tính toán truyền thống.")
    st.info(f"Kiểm tra các file '{PREPROCESSOR_PATH}' và '{FEATURES_PATH}'.")

# --- Chân trang ---
st.markdown("---")
st.caption("Ứng dụng demo xây dựng với Streamlit, Scikit-learn, XGBoost và các mô hình truyền thống.")