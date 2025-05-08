# -*- coding: utf-8 -*-
"""
app.py: Ứng dụng Web Streamlit để dự đoán Effort, Thời gian và Quy mô đội
(Phiên bản có chọn ngôn ngữ cho LOC/FP, ẩn ML extras nếu không cần).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import OrderedDict
import math  # Cần cho COCOMO
import traceback  # Thêm để in lỗi chi tiết

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

# --- Hằng số cho năng suất mặc định ---
HOURS_PER_FP_DEFAULT = 10.0
HOURS_PER_UCP_DEFAULT = 20.0

# --- Định nghĩa các Ngôn ngữ và LOC/FP tương ứng ---
LANGUAGE_LOC_FP_MAP = OrderedDict([
    ("Java (3GL)", 53),
    ("C# (3GL)", 54),
    ("C++ (3GL)", 60),  # Thêm ví dụ
    ("Python (3GL/Scripting)", 35),
    ("JavaScript (3GL/Scripting)", 47),
    ("PHP (Scripting)", 40),  # Thêm ví dụ
    ("SQL (4GL)", 15),
    ("Oracle Forms (4GL)", 20),
    ("PowerBuilder (4GL)", 16),  # Thêm ví dụ
    ("Trung bình 3GL", 65),  # Giá trị tham khảo chung
    ("Trung bình 4GL", 20),  # Giá trị tham khảo chung
    ("Tùy chỉnh", None)  # Cho phép nhập tay
])

# --- Cấu hình Trang và Tải Artifacts ---
st.set_page_config(page_title="Ước tính Effort, Thời gian & Đội ngũ Phần mềm", layout="wide")

st.title("Ứng dụng So sánh Ước tính Effort, Thời gian & Đội ngũ Phần mềm 📊")
st.write("""
Nhập thông tin dự án để nhận ước tính effort (person-hours), thời gian phát triển (tháng)
và quy mô đội ngũ từ nhiều mô hình.
""")

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
    all_loaded_successfully = True

    if not os.path.exists(preprocessor_path):
        st.error(f"LỖI: Không tìm thấy file preprocessor tại '{preprocessor_path}'")
        return None, None, None, None, None
    try:
        preprocessor = joblib.load(preprocessor_path)
        try:
            num_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'num')
            cat_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'cat')
            original_num_features = list(num_transformer_tuple[2])
            original_cat_features = list(cat_transformer_tuple[2])
            original_cols_order = original_num_features + original_cat_features

            cat_pipeline = preprocessor.named_transformers_['cat']
            onehot_encoder = cat_pipeline.named_steps['onehot']

            if hasattr(onehot_encoder, 'categories_'):
                if len(onehot_encoder.categories_) == len(original_cat_features):
                    for i, feature_name in enumerate(original_cat_features):
                        categories = onehot_encoder.categories_[i]
                        categorical_features_options[feature_name] = categories.tolist()
                else:
                    all_loaded_successfully = False
            else:
                all_loaded_successfully = False
        except Exception:
            all_loaded_successfully = False
    except Exception:
        return None, None, None, None, None

    if not os.path.exists(features_path):
        all_loaded_successfully = False
    else:
        try:
            feature_names = joblib.load(features_path)
            if isinstance(feature_names, np.ndarray): feature_names = feature_names.tolist()
            if not isinstance(feature_names, list):
                try:
                    feature_names = list(feature_names)
                except TypeError:
                    all_loaded_successfully = False
        except Exception:
            all_loaded_successfully = False

    models_actually_loaded = 0
    for name, path in model_paths_dict.items():
        if not os.path.exists(path): continue
        try:
            model = joblib.load(path)
            loaded_models[name] = model
            models_actually_loaded += 1
        except Exception:
            pass

    if all_loaded_successfully and preprocessor and feature_names and original_cols_order and categorical_features_options:
        return preprocessor, feature_names, loaded_models, original_cols_order, categorical_features_options
    else:
        return preprocessor, feature_names, loaded_models, original_cols_order, categorical_features_options


preprocessor, feature_names_out, ml_models, original_cols_order, categorical_features_options = load_all_artifacts(
    PREPROCESSOR_PATH, FEATURES_PATH, MODEL_PATHS
)


def calculate_cocomo_basic(loc, mode, eaf, hrs_per_month_cocomo):
    if loc <= 0: return "Lỗi (LOC <= 0)"
    if hrs_per_month_cocomo <= 0: return "Lỗi (Giờ/Tháng COCOMO <= 0)"
    kloc = loc / 1000.0
    params = {"Organic": {"a": 2.4, "b": 1.05}, "Semi-detached": {"a": 3.0, "b": 1.12},
              "Embedded": {"a": 3.6, "b": 1.20}}
    if mode not in params: return "Lỗi (Chế độ không hợp lệ)"
    a, b = params[mode]["a"], params[mode]["b"]
    try:
        person_months = a * (kloc ** b) * eaf
        person_hours = person_months * hrs_per_month_cocomo
        return max(0.0, round(person_hours, 2))
    except Exception as e:
        return f"Lỗi tính toán COCOMO: {e}"


def calculate_fp_effort(fp):
    if fp <= 0: return "Lỗi (FP <= 0)"
    if HOURS_PER_FP_DEFAULT <= 0: return "Lỗi (Năng suất FP mặc định <= 0)"
    try:
        person_hours = fp * HOURS_PER_FP_DEFAULT
        return max(0.0, round(person_hours, 2))
    except Exception as e:
        return f"Lỗi tính toán FP: {e}"


def calculate_ucp_effort(ucp):
    if ucp <= 0: return "Lỗi (UCP <= 0)"
    if HOURS_PER_UCP_DEFAULT <= 0: return "Lỗi (Năng suất UCP mặc định <= 0)"
    try:
        person_hours = ucp * HOURS_PER_UCP_DEFAULT
        return max(0.0, round(person_hours, 2))
    except Exception as e:
        return f"Lỗi tính toán UCP: {e}"


st.sidebar.header("📝 Nhập Thông tin Dự án")
input_values_ml_extra = {}

st.sidebar.subheader("📏 Chỉ số Kích thước & Chuyển đổi")
primary_metric_type = st.sidebar.radio(
    "Chọn chỉ số đầu vào chính:",
    ("LOC", "FP", "UCP"), key="primary_metric_type", horizontal=True
)

primary_metric_val = 0
if primary_metric_type == "LOC":
    primary_metric_val = st.sidebar.number_input("Giá trị Lines of Code (LOC):", min_value=0, value=10000, step=100,
                                                 key="loc_primary_input")
elif primary_metric_type == "FP":
    primary_metric_val = st.sidebar.number_input("Giá trị Function Points (FP):", min_value=0, value=100, step=10,
                                                 key="fp_primary_input")
else:
    primary_metric_val = st.sidebar.number_input("Giá trị Use Case Points (UCP):", min_value=0.0, value=100.0,
                                                 step=10.0, format="%.2f", key="ucp_primary_input")

st.sidebar.markdown("Hệ số chuyển đổi LOC/FP:")
selected_language_profile = st.sidebar.selectbox(
    "Ngôn ngữ lập trình / Loại (để ước tính LOC/FP):",
    options=list(LANGUAGE_LOC_FP_MAP.keys()),
    index=0,  # Mặc định chọn cái đầu tiên
    key="lang_profile"
)

loc_per_fp_factor_to_use = LANGUAGE_LOC_FP_MAP[selected_language_profile]
if loc_per_fp_factor_to_use is None:  # Trường hợp "Tùy chỉnh"
    loc_per_fp_factor_to_use = st.sidebar.number_input(
        "Nhập Số LOC trung bình / 1 FP:",
        min_value=1.0, value=50.0, step=1.0, format="%.1f", key="loc_per_fp_manual"
    )
else:
    st.sidebar.text_input(
        "Số LOC trung bình / 1 FP (từ ngôn ngữ):",
        value=f"{loc_per_fp_factor_to_use}",
        disabled=True
    )

fp_per_ucp_factor = st.sidebar.number_input(
    "Số FP trung bình / 1 UCP (ví dụ: 1.5 - 3.0):",
    min_value=0.1, value=2.0, step=0.1, format="%.1f", key="fp_per_ucp"
)

# --- 2. Widget nhập liệu chỉ cho ML (Đặc trưng bổ sung - chỉ hiển thị nếu cần) ---
# Xác định xem có đặc trưng ML bổ sung nào cần hiển thị không
supplementary_numeric_ml_features_needed = []
supplementary_categorical_ml_features_needed = []

if preprocessor and original_cols_order:
    # Các đặc trưng số bổ sung tiềm năng (không phải LOC/FP/UCP)
    potential_numeric_extras = ['Development Time (months)', 'Team Size']  # Thêm các feature khác nếu có
    for feat_name in potential_numeric_extras:
        if feat_name in original_cols_order and (
                categorical_features_options is None or feat_name not in categorical_features_options.keys()):
            supplementary_numeric_ml_features_needed.append(feat_name)

    # Các đặc trưng phân loại bổ sung tiềm năng (ví dụ: 'Project Type', 'Complexity Level' nếu ML model dùng)
    # Giả sử original_cols_order và categorical_features_options đã đúng từ preprocessor
    if categorical_features_options:
        for cat_feat_name in categorical_features_options.keys():
            if cat_feat_name in original_cols_order:  # Đảm bảo feature này thực sự được preprocessor sử dụng
                # Quyết định xem feature này có phải là "bổ sung" hay không
                # Ví dụ, nếu bạn có 1 feature 'Deployment Environment' mà không phải là 1 trong các input chính khác
                # if cat_feat_name not in ['Some_Primary_Categorical_Input_Handled_Elsewhere']:
                supplementary_categorical_ml_features_needed.append(cat_feat_name)

show_ml_extra_section = bool(supplementary_numeric_ml_features_needed or supplementary_categorical_ml_features_needed)

if show_ml_extra_section:
    st.sidebar.subheader("⚙️ Đặc trưng Bổ sung (Cho ML nếu mô hình yêu cầu)")
    if supplementary_numeric_ml_features_needed:
        col_ml_num1, col_ml_num2 = st.sidebar.columns(2)
        # Hiển thị các input số bổ sung
        # Ví dụ cho 'Development Time (months)' và 'Team Size' nếu chúng nằm trong supplementary_numeric_ml_features_needed
        # Cần làm cho phần này linh động hơn nếu có nhiều feature số bổ sung
        with col_ml_num1:
            if 'Development Time (months)' in supplementary_numeric_ml_features_needed:
                input_values_ml_extra['Development Time (months)'] = st.number_input(
                    "Dev Time (tháng) (ML Feature)", min_value=1, value=6, step=1, key="ml_dev_time_feature_input"
                )
        with col_ml_num2:
            if 'Team Size' in supplementary_numeric_ml_features_needed:
                input_values_ml_extra['Team Size'] = st.number_input(
                    "Team Size (người) (ML Feature)", min_value=1, value=5, step=1, key="ml_team_size_feature_input"
                )
        # Thêm các input số khác tương tự nếu cần

    if supplementary_categorical_ml_features_needed:
        st.sidebar.markdown("**Thông tin Phân loại Bổ sung (Cho ML):**")  # Tiêu đề con nếu có
        col_cat_ml1, col_cat_ml2 = st.sidebar.columns(2)
        half_len_cat_extra = (len(supplementary_categorical_ml_features_needed) + 1) // 2

        current_cat_list = supplementary_categorical_ml_features_needed  # Sử dụng list đã lọc

        with col_cat_ml1:
            for i, col_name in enumerate(current_cat_list[:half_len_cat_extra]):
                if col_name in categorical_features_options:  # Double check
                    options = categorical_features_options[col_name]
                    input_values_ml_extra[col_name] = st.selectbox(
                        f"{col_name}", options=options, index=0, key=f"sb_extra_{col_name}_1_ml"
                    )
        with col_cat_ml2:
            for i, col_name in enumerate(current_cat_list[half_len_cat_extra:]):
                if col_name in categorical_features_options:  # Double check
                    options = categorical_features_options[col_name]
                    input_values_ml_extra[col_name] = st.selectbox(
                        f"{col_name}", options=options, index=0, key=f"sb_extra_{col_name}_2_ml"
                    )
# else:
# st.sidebar.info("Không có đặc trưng ML bổ sung nào được yêu cầu bởi mô hình đã tải.")


st.sidebar.subheader("📜 Tham số cho COCOMO II (Basic)")
cocomo_mode = st.sidebar.selectbox("Chế độ Dự án COCOMO", ["Organic", "Semi-detached", "Embedded"], key="cocomo_mode")
eaf = st.sidebar.number_input("Hệ số Điều chỉnh Nỗ lực COCOMO (EAF)", min_value=0.1, value=1.0, step=0.1, format="%.2f",
                              key="eaf")
hours_per_month_cocomo = st.sidebar.number_input("Số giờ/tháng (COCOMO PM to PH)", min_value=1, value=152, step=8,
                                                 key="hrs_month_cocomo")

st.sidebar.subheader("⏱️ Ước tính Thời gian & Đội ngũ (Lập kế hoạch)")
scheduling_basis = st.sidebar.radio(
    "Tính toán dựa trên:",
    ("Quy mô đội ngũ đã biết", "Thời gian phát triển mong muốn"), key="scheduling_basis", horizontal=True
)
team_size_input_sched = None
dev_time_input_sched = None
if scheduling_basis == "Quy mô đội ngũ đã biết":
    team_size_input_sched = st.sidebar.number_input("Quy mô đội ngũ (số người)", min_value=1, value=5, step=1,
                                                    key="team_size_for_sched")
else:
    dev_time_input_sched = st.sidebar.number_input("Thời gian phát triển mong muốn (tháng)", min_value=1.0, value=6.0,
                                                   step=0.5, format="%.1f", key="dev_time_for_sched")

effective_hours_per_month_sched = st.sidebar.number_input(
    "Số giờ làm việc hiệu quả/người/tháng (cho lập kế hoạch)",
    min_value=1, value=140, step=8, key="eff_hrs_month_sched",
    help="Ví dụ: 8 giờ/ngày * 20 ngày/tháng * 0.875 (hiệu suất) = 140 giờ"
)

calculate_button = st.sidebar.button("🚀 Ước tính & So sánh", use_container_width=True, type="primary")

if calculate_button:
    st.divider()
    st.subheader("📊 Kết quả Ước tính Tổng hợp")

    loc_to_use, fp_to_use, ucp_to_use = 0.0, 0.0, 0.0
    conversion_errors = []

    try:
        if loc_per_fp_factor_to_use <= 0: conversion_errors.append("Hệ số LOC/FP phải > 0.")
        if fp_per_ucp_factor <= 0: conversion_errors.append("Hệ số FP/UCP phải > 0.")

        if not conversion_errors:
            if primary_metric_type == "LOC":
                loc_to_use = float(primary_metric_val)
                fp_to_use = loc_to_use / loc_per_fp_factor_to_use if loc_per_fp_factor_to_use > 0 else 0.0
                ucp_to_use = fp_to_use / fp_per_ucp_factor if fp_per_ucp_factor > 0 else 0.0
            elif primary_metric_type == "FP":
                fp_to_use = float(primary_metric_val)
                loc_to_use = fp_to_use * loc_per_fp_factor_to_use
                ucp_to_use = fp_to_use / fp_per_ucp_factor if fp_per_ucp_factor > 0 else 0.0
            else:  # UCP
                ucp_to_use = float(primary_metric_val)
                fp_to_use = ucp_to_use * fp_per_ucp_factor
                loc_to_use = fp_to_use * loc_per_fp_factor_to_use

        loc_to_use_display = round(loc_to_use)
        fp_to_use_display = round(fp_to_use, 2)
        ucp_to_use_display = round(ucp_to_use, 2)

    except Exception as e_conv:
        conversion_errors.append(f"Lỗi trong quá trình chuyển đổi chỉ số: {e_conv}")

    if conversion_errors:
        for err in conversion_errors: st.error(f"Lỗi hệ số chuyển đổi: {err}")
        st.stop()

    st.markdown("#### Giá trị kích thước được sử dụng cho tính toán:")
    col_size1, col_size2, col_size3 = st.columns(3)
    col_size1.metric("Lines of Code (LOC)", f"{loc_to_use_display:,.0f}")
    col_size2.metric("Function Points (FP)", f"{fp_to_use_display:,.2f}")
    col_size3.metric("Use Case Points (UCP)", f"{ucp_to_use_display:,.2f}")
    st.caption(
        f"(Đầu vào chính: {primary_metric_type}. Ngôn ngữ/Loại cho LOC/FP: {selected_language_profile} -> {loc_per_fp_factor_to_use} LOC/FP.)")
    st.markdown("---")

    all_results_list = []
    error_messages_ml = {}

    # --- 1. Dự đoán từ Mô hình Machine Learning ---
    if preprocessor and feature_names_out and ml_models and original_cols_order and categorical_features_options:
        st.markdown("##### 1. Dự đoán từ Mô hình Machine Learning")
        try:
            current_input_data_ml = input_values_ml_extra.copy()
            if 'LOC' in original_cols_order: current_input_data_ml['LOC'] = loc_to_use
            if 'FP' in original_cols_order: current_input_data_ml['FP'] = fp_to_use
            if 'UCP' in original_cols_order: current_input_data_ml['UCP'] = ucp_to_use
            # Nếu ML model được huấn luyện với feature 'Language Profile' hoặc tương tự,
            # bạn cần thêm selected_language_profile vào current_input_data_ml ở đây.
            # Ví dụ: if 'Language_Profile_Feature_Name_In_Model' in original_cols_order:
            # current_input_data_ml['Language_Profile_Feature_Name_In_Model'] = selected_language_profile

            ordered_input_data_for_ml_df = {}
            missing_inputs_ml = []
            for col in original_cols_order:
                if col in current_input_data_ml:
                    ordered_input_data_for_ml_df[col] = current_input_data_ml[col]
                else:
                    missing_inputs_ml.append(col)
                    ordered_input_data_for_ml_df[col] = np.nan

            if missing_inputs_ml:
                st.warning(f"ML Input: Thiếu giá trị cho: {', '.join(missing_inputs_ml)}. Sẽ được Imputer xử lý.")

            input_df_ml = pd.DataFrame([ordered_input_data_for_ml_df], columns=original_cols_order)
            input_processed_np = preprocessor.transform(input_df_ml)

            if isinstance(feature_names_out, list) and len(feature_names_out) == input_processed_np.shape[1]:
                input_processed_df = pd.DataFrame(input_processed_np, columns=feature_names_out)
                for model_name, loaded_model in ml_models.items():
                    effort_ph, dev_time_m, team_size_p = "Lỗi", "N/A", "N/A"
                    try:
                        pred = loaded_model.predict(input_processed_df)
                        effort_ph = max(0.0, round(float(pred[0]) if pred.size > 0 else 0.0, 2))
                        if isinstance(effort_ph, float) and effective_hours_per_month_sched > 0:
                            effort_pm_sched = effort_ph / effective_hours_per_month_sched
                            if scheduling_basis == "Quy mô đội ngũ đã biết" and team_size_input_sched and team_size_input_sched > 0:
                                dev_time_m = round(effort_pm_sched / team_size_input_sched, 1)
                                team_size_p = team_size_input_sched
                            elif scheduling_basis == "Thời gian phát triển mong muốn" and dev_time_input_sched and dev_time_input_sched > 0:
                                team_size_p = math.ceil(effort_pm_sched / dev_time_input_sched)
                                dev_time_m = dev_time_input_sched
                    except Exception as model_pred_e:
                        error_messages_ml[model_name] = str(model_pred_e)
                    all_results_list.append({
                        'LOC': loc_to_use_display, 'UCP': ucp_to_use_display, 'FP': fp_to_use_display,
                        'Mô Hình': f"ML: {model_name}", 'Effort (giờ)': effort_ph,
                        'Thời Gian (Tháng)': dev_time_m, 'Đội Ngũ (Người)': team_size_p
                    })
            else:
                st.error(f"Lỗi ML: Số đặc trưng ({len(feature_names_out)}) không khớp ({input_processed_np.shape[1]}).")
                for model_name in ml_models.keys():
                    all_results_list.append(
                        {'LOC': loc_to_use_display, 'UCP': ucp_to_use_display, 'FP': fp_to_use_display,
                         'Mô Hình': f"ML: {model_name}", 'Effort (giờ)': "Lỗi (Config)", 'Thời Gian (Tháng)': "N/A",
                         'Đội Ngũ (Người)': "N/A"})
        except Exception as e_ml_process:
            st.error(f"Lỗi xử lý/dự đoán ML: {e_ml_process}")
            for model_name in ml_models.keys():
                all_results_list.append({'LOC': loc_to_use_display, 'UCP': ucp_to_use_display, 'FP': fp_to_use_display,
                                         'Mô Hình': f"ML: {model_name}", 'Effort (giờ)': "Lỗi (Process)",
                                         'Thời Gian (Tháng)': "N/A", 'Đội Ngũ (Người)': "N/A"})
            print(traceback.format_exc())
    else:
        st.info("Dự đoán ML không thực hiện do thiếu thành phần / không có model ML.")

    # --- 2. Tính toán từ Mô hình Truyền thống ---
    st.markdown("##### 2. Tính toán từ Mô hình Truyền thống")
    traditional_models_params_display = []

    effort_cocomo = calculate_cocomo_basic(loc_to_use, cocomo_mode, eaf, hours_per_month_cocomo)
    dt_cocomo, ts_cocomo = "N/A", "N/A"
    if isinstance(effort_cocomo, float) and effective_hours_per_month_sched > 0:
        effort_pm_sched_cocomo = effort_cocomo / effective_hours_per_month_sched
        if scheduling_basis == "Quy mô đội ngũ đã biết" and team_size_input_sched and team_size_input_sched > 0:
            dt_cocomo = round(effort_pm_sched_cocomo / team_size_input_sched, 1);
            ts_cocomo = team_size_input_sched
        elif scheduling_basis == "Thời gian phát triển mong muốn" and dev_time_input_sched and dev_time_input_sched > 0:
            ts_cocomo = math.ceil(effort_pm_sched_cocomo / dev_time_input_sched);
            dt_cocomo = dev_time_input_sched
    all_results_list.append(
        {'LOC': loc_to_use_display, 'UCP': ucp_to_use_display, 'FP': fp_to_use_display, 'Mô Hình': 'COCOMO II (Basic)',
         'Effort (giờ)': effort_cocomo, 'Thời Gian (Tháng)': dt_cocomo, 'Đội Ngũ (Người)': ts_cocomo})
    traditional_models_params_display.append(
        f"* **COCOMO II:** Mode={cocomo_mode}, EAF={eaf}, Hrs/Month (COCOMO)={hours_per_month_cocomo}")

    effort_fp = calculate_fp_effort(fp_to_use)
    dt_fp, ts_fp = "N/A", "N/A"
    if isinstance(effort_fp, float) and effective_hours_per_month_sched > 0:
        effort_pm_sched_fp = effort_fp / effective_hours_per_month_sched
        if scheduling_basis == "Quy mô đội ngũ đã biết" and team_size_input_sched and team_size_input_sched > 0:
            dt_fp = round(effort_pm_sched_fp / team_size_input_sched, 1);
            ts_fp = team_size_input_sched
        elif scheduling_basis == "Thời gian phát triển mong muốn" and dev_time_input_sched and dev_time_input_sched > 0:
            ts_fp = math.ceil(effort_pm_sched_fp / dev_time_input_sched);
            dt_fp = dev_time_input_sched
    all_results_list.append(
        {'LOC': loc_to_use_display, 'UCP': ucp_to_use_display, 'FP': fp_to_use_display, 'Mô Hình': 'Function Points',
         'Effort (giờ)': effort_fp, 'Thời Gian (Tháng)': dt_fp, 'Đội Ngũ (Người)': ts_fp})
    traditional_models_params_display.append(f"* **Function Points:** Năng suất mặc định={HOURS_PER_FP_DEFAULT} giờ/FP")

    effort_ucp = calculate_ucp_effort(ucp_to_use)
    dt_ucp, ts_ucp = "N/A", "N/A"
    if isinstance(effort_ucp, float) and effective_hours_per_month_sched > 0:
        effort_pm_sched_ucp = effort_ucp / effective_hours_per_month_sched
        if scheduling_basis == "Quy mô đội ngũ đã biết" and team_size_input_sched and team_size_input_sched > 0:
            dt_ucp = round(effort_pm_sched_ucp / team_size_input_sched, 1);
            ts_ucp = team_size_input_sched
        elif scheduling_basis == "Thời gian phát triển mong muốn" and dev_time_input_sched and dev_time_input_sched > 0:
            ts_ucp = math.ceil(effort_pm_sched_ucp / dev_time_input_sched);
            dt_ucp = dev_time_input_sched
    all_results_list.append(
        {'LOC': loc_to_use_display, 'UCP': ucp_to_use_display, 'FP': fp_to_use_display, 'Mô Hình': 'Use Case Points',
         'Effort (giờ)': effort_ucp, 'Thời Gian (Tháng)': dt_ucp, 'Đội Ngũ (Người)': ts_ucp})
    traditional_models_params_display.append(
        f"* **Use Case Points:** Năng suất mặc định={HOURS_PER_UCP_DEFAULT} giờ/UCP")

    st.markdown("**Tham số mô hình truyền thống (ngoài LOC/FP/UCP):**")
    for caption in traditional_models_params_display: st.markdown(caption)
    st.markdown("**Tham số lập kế hoạch chung:**")
    st.markdown(f"* Tính toán dựa trên: **{scheduling_basis}**")
    if scheduling_basis == "Quy mô đội ngũ đã biết":
        st.markdown(f"* Quy mô đội cung cấp: **{team_size_input_sched or 'N/A'} người**")
    else:
        st.markdown(f"* Thời gian phát triển mong muốn: **{dev_time_input_sched or 'N/A'} tháng**")
    st.markdown(f"* Số giờ hiệu quả/người/tháng: **{effective_hours_per_month_sched} giờ**")
    st.caption("Lưu ý: 'Lỗi' nếu đầu vào không hợp lệ. 'N/A' nếu không thể tính thời gian/đội ngũ.")
    st.markdown("---")

    st.markdown("##### 3. Bảng và Biểu đồ So sánh")
    if all_results_list:
        result_df = pd.DataFrame(all_results_list)
        cols_ordered = ['LOC', 'UCP', 'FP', 'Mô Hình', 'Effort (giờ)', 'Thời Gian (Tháng)', 'Đội Ngũ (Người)']
        result_df = result_df[cols_ordered]


        def format_value_display(value, is_effort_or_loc=False, is_fp_ucp=False):
            if isinstance(value, (int, float)):
                if np.isnan(value): return "N/A"
                if is_effort_or_loc:
                    return f"{value:,.0f}" if value % 1 == 0 else f"{value:,.2f}"
                elif is_fp_ucp:
                    return f"{value:,.2f}"
                else:
                    return f"{value:,.1f}" if value % 1 != 0 else f"{value:,.0f}"
            return str(value)


        st.write("Bảng so sánh kết quả:")
        display_df = result_df.copy()
        display_df['LOC'] = display_df['LOC'].apply(lambda x: format_value_display(x, is_effort_or_loc=True))
        display_df['UCP'] = display_df['UCP'].apply(lambda x: format_value_display(x, is_fp_ucp=True))
        display_df['FP'] = display_df['FP'].apply(lambda x: format_value_display(x, is_fp_ucp=True))
        display_df['Effort (giờ)'] = display_df['Effort (giờ)'].apply(
            lambda x: format_value_display(x, is_effort_or_loc=True))
        display_df['Thời Gian (Tháng)'] = display_df['Thời Gian (Tháng)'].apply(format_value_display)
        display_df['Đội Ngũ (Người)'] = display_df['Đội Ngũ (Người)'].apply(format_value_display)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.write("Biểu đồ so sánh Effort (giờ):")
        try:
            chart_df_effort = result_df.copy()
            chart_df_effort['Effort (giờ)'] = pd.to_numeric(chart_df_effort['Effort (giờ)'], errors='coerce')
            chart_df_effort.dropna(subset=['Effort (giờ)'], inplace=True)
            if not chart_df_effort.empty:
                st.bar_chart(chart_df_effort.set_index('Mô Hình')['Effort (giờ)'])
            else:
                st.info("Không có dữ liệu Effort hợp lệ để vẽ biểu đồ.")
        except Exception as chart_e:
            st.warning(f"Không thể vẽ biểu đồ Effort: {chart_e}"); print(traceback.format_exc())
    else:
        st.warning("Không có kết quả nào để hiển thị.")

    if error_messages_ml:
        st.subheader("⚠️ Chi tiết lỗi dự đoán ML:")
        for model_name, msg in error_messages_ml.items(): st.caption(f"**{model_name}:** {msg}")
    st.info("""**Lưu ý quan trọng:** Kết quả chỉ là **ước tính**. Độ chính xác phụ thuộc vào nhiều yếu tố.""")

elif not ml_models and not preprocessor and not os.path.exists(PREPROCESSOR_PATH):
    st.error("Lỗi tải thành phần ML (preprocessor, models). Kiểm tra đường dẫn.")
elif not ml_models and (preprocessor or os.path.exists(PREPROCESSOR_PATH)):
    st.warning("Không tải được model ML. Phần ML sẽ không hoạt động.")
elif not preprocessor or not feature_names_out or not original_cols_order or not categorical_features_options:
    st.error("Lỗi tải/xử lý preprocessor/thông tin đặc trưng ML.")

st.markdown("---")
st.caption(
    f"Demo App. Năng suất FP mặc định: {HOURS_PER_FP_DEFAULT} giờ/FP. Năng suất UCP mặc định: {HOURS_PER_UCP_DEFAULT} giờ/UCP.")