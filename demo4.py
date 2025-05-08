# -*- coding: utf-8 -*-
"""
app.py: Ứng dụng Web Streamlit để dự đoán Effort
(Bao gồm Mô hình ML, COCOMO II Basic, FP, UCP và So sánh).
Phiên bản nâng cấp: Quy đổi LOC, FP, UCP và tính toán TDEV, Team Size.
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

# --- Cấu hình Trang ---
st.set_page_config(page_title="So sánh Ước tính Effort Phần mềm", layout="wide")

st.title("Ứng dụng So sánh Ước tính Effort Phần mềm 📊")
st.write("""
Nhập thông tin dự án để nhận ước tính effort (person-hours), thời gian phát triển, và kích thước đội ngũ
từ nhiều mô hình Machine Learning và các phương pháp truyền thống.
""")

# --- Định nghĩa Đường dẫn và Hằng số ---
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

# Hằng số cho quy đổi và COCOMO
LOC_PER_FP_LANGUAGES = OrderedDict([
    ("3GL (Java, C#, C++)", {"avg": 55}),  # Giá trị ví dụ, cần điều chỉnh theo ngữ cảnh
    ("4GL (SQL, RAD tools)", {"avg": 20}),
    ("Python", {"avg": 40}),
    ("JavaScript (Frontend)", {"avg": 50}),
    ("Ngôn ngữ khác/Trung bình", {"avg": 50})  # Mặc định
])
DEFAULT_LOC_PER_FP = LOC_PER_FP_LANGUAGES["Ngôn ngữ khác/Trung bình"]["avg"]

# Tỷ lệ quy đổi UCP và FP (ví dụ: 1 UCP ~ 1.5-2.5 FP)
# Giả sử 1 UCP = 2 FP (UCP thường lớn hơn FP về scope)
UCP_TO_FP_RATIO = 2.0
FP_TO_UCP_RATIO = 1.0 / UCP_TO_FP_RATIO

# Tham số COCOMO II (Basic Effort + TDEV)
# Effort (PM) = a * (KLOC)^b * EAF
# TDEV (Months) = c * (PM_EAF_Adjusted)^d
# Lưu ý: EAF được áp dụng cho Effort (PM), sau đó PM đã điều chỉnh được dùng cho TDEV
COCOMO_PARAMS = {
    "Organic": {"a": 2.4, "b": 1.05, "c": 2.5, "d": 0.38},
    "Semi-detached": {"a": 3.0, "b": 1.12, "c": 2.5, "d": 0.35},
    "Embedded": {"a": 3.6, "b": 1.20, "c": 2.5, "d": 0.32}
}


# --- Tải Artifacts ---
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
        return None, None, None, None, None, False  # Thêm cờ trạng thái
    try:
        preprocessor = joblib.load(preprocessor_path)
        num_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'num')
        cat_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'cat')
        original_num_features = list(num_transformer_tuple[2])
        original_cat_features = list(cat_transformer_tuple[2])
        original_cols_order = original_num_features + original_cat_features

        cat_pipeline = preprocessor.named_transformers_['cat']
        onehot_encoder = cat_pipeline.named_steps['onehot']

        if hasattr(onehot_encoder, 'categories_') and len(onehot_encoder.categories_) == len(original_cat_features):
            for i, feature_name in enumerate(original_cat_features):
                categorical_features_options[feature_name] = onehot_encoder.categories_[i].tolist()
        else:
            st.error("Lỗi trích xuất danh mục từ OneHotEncoder.")
            all_loaded_successfully = False
    except Exception as e_load_prep:
        st.error(f"Lỗi nghiêm trọng khi tải preprocessor: {e_load_prep}")
        return None, None, None, None, None, False

    if not os.path.exists(features_path):
        st.error(f"LỖI: Không tìm thấy file tên đặc trưng tại '{features_path}'")
        all_loaded_successfully = False
    else:
        try:
            feature_names = joblib.load(features_path)
            if isinstance(feature_names, np.ndarray): feature_names = feature_names.tolist()
        except Exception as e_load_feat:
            st.error(f"Lỗi khi tải feature names: {e_load_feat}")
            all_loaded_successfully = False

    models_actually_loaded = 0
    for name, path in model_paths_dict.items():
        if not os.path.exists(path):
            st.warning(f"Cảnh báo: Không tìm thấy file mô hình '{name}' tại '{path}'. Bỏ qua.")
            continue
        try:
            loaded_models[name] = joblib.load(path)
            models_actually_loaded += 1
        except Exception as e_load_model:
            st.warning(f"Lỗi khi tải mô hình {name}: {e_load_model}. Bỏ qua.")

    if models_actually_loaded == 0:
        st.warning("Không tải được bất kỳ mô hình Machine Learning nào.")
        # all_loaded_successfully = False # Vẫn có thể chạy mô hình truyền thống

    return preprocessor, feature_names, loaded_models, original_cols_order, categorical_features_options, all_loaded_successfully


preprocessor, feature_names_out, ml_models, original_cols_order, categorical_features_options, artifacts_loaded_successfully = load_all_artifacts(
    PREPROCESSOR_PATH, FEATURES_PATH, MODEL_PATHS
)


# --- Hàm Tính toán cho Mô hình Truyền thống và Quy đổi ---

def get_loc_per_fp(language_choice_key):
    """Lấy tỷ lệ LOC/FP dựa trên lựa chọn ngôn ngữ."""
    return LOC_PER_FP_LANGUAGES.get(language_choice_key, {"avg": DEFAULT_LOC_PER_FP})["avg"]


def convert_software_sizes(value, input_type, loc_per_fp_rate, ucp_fp_ratio, fp_ucp_ratio):
    """
    Quy đổi giữa LOC, FP, UCP.
    Trả về một dictionary: {'loc': loc_val, 'fp': fp_val, 'ucp': ucp_val}
    """
    loc_val, fp_val, ucp_val = 0, 0, 0
    if value is None or value < 0: value = 0  # Xử lý giá trị không hợp lệ

    try:
        if input_type == 'LOC':
            loc_val = float(value)
            fp_val = loc_val / loc_per_fp_rate if loc_per_fp_rate > 0 else 0
            ucp_val = fp_val * fp_ucp_ratio  # FP nhỏ hơn UCP, nên UCP = FP / (FP/UCP_ratio) = FP * (UCP/FP_ratio)
        elif input_type == 'FP':
            fp_val = float(value)
            loc_val = fp_val * loc_per_fp_rate
            ucp_val = fp_val * fp_ucp_ratio
        elif input_type == 'UCP':
            ucp_val = float(value)
            fp_val = ucp_val * ucp_fp_ratio  # UCP to FP
            loc_val = fp_val * loc_per_fp_rate
        else:
            return {'loc': 0, 'fp': 0, 'ucp': 0, 'error': "Loại input không hợp lệ"}

        return {
            'loc': round(loc_val, 0),
            'fp': round(fp_val, 2),
            'ucp': round(ucp_val, 2),
            'error': None
        }
    except Exception as e:
        return {'loc': 0, 'fp': 0, 'ucp': 0, 'error': str(e)}


def calculate_cocomo_ii_extended(loc, mode, eaf, hrs_per_month, params):
    """
    Tính toán Effort (PM, Person-hours), TDEV (tháng), và Staff (người) theo COCOMO II.
    """
    results = {
        'effort_pm': "N/A", 'effort_hrs': "N/A",
        'tdev_months': "N/A", 'avg_staff': "N/A", 'error': None
    }
    if loc <= 0:
        results['error'] = "LOC phải > 0"
        return results
    if hrs_per_month <= 0:
        results['error'] = "Số giờ/tháng phải > 0"
        return results
    if mode not in params:
        results['error'] = "Chế độ COCOMO không hợp lệ"
        return results

    kloc = loc / 1000.0
    mode_params = params[mode]
    a, b, c, d = mode_params["a"], mode_params["b"], mode_params["c"], mode_params["d"]

    try:
        # Effort (Person-Months) = a * (KLOC)^b * EAF
        effort_pm = a * (kloc ** b) * eaf
        results['effort_pm'] = round(effort_pm, 2)

        # Effort (Person-Hours)
        effort_hrs = effort_pm * hrs_per_month
        results['effort_hrs'] = round(effort_hrs, 2)

        # TDEV (Development Time in Months) = c * (Effort_PM_Adjusted_by_EAF)^d
        # EAF đã được tính vào effort_pm
        tdev_months = c * (effort_pm ** d)
        results['tdev_months'] = round(tdev_months, 2)

        # Average Staffing (Persons) = Effort_PM / TDEV_Months
        if tdev_months > 0:
            avg_staff = effort_pm / tdev_months
            results['avg_staff'] = round(avg_staff, 2)
        else:
            results['avg_staff'] = "N/A (TDEV <=0)"

        return results
    except Exception as e:
        results['error'] = f"Lỗi tính toán COCOMO: {e}"
        return results


def calculate_fp_effort(fp, hrs_per_fp):
    if fp <= 0 or hrs_per_fp <= 0: return "Lỗi (FP hoặc Giờ/FP <= 0)"
    try:
        return round(fp * hrs_per_fp, 2)
    except:
        return "Lỗi tính toán"


def calculate_ucp_effort(ucp, hrs_per_ucp):
    if ucp <= 0 or hrs_per_ucp <= 0: return "Lỗi (UCP hoặc Giờ/UCP <= 0)"
    try:
        return round(ucp * hrs_per_ucp, 2)
    except:
        return "Lỗi tính toán"


# --- Giao diện Nhập liệu Sidebar ---
st.sidebar.header("Nhập Thông tin Dự án")

# Mục mới cho Quy đổi và Ước tính Thời gian/Nhân lực
st.sidebar.subheader("Quy đổi Kích thước & Ước tính T.Gian/Nhân lực")
primary_metric_for_conversion = st.sidebar.radio(
    "Chọn Metric chính để nhập và quy đổi:",
    ('LOC', 'FP', 'UCP'),
    key='primary_metric_source',
    horizontal=True,
    index=0  # Mặc định chọn LOC
)

language_for_conversion_key = st.sidebar.selectbox(
    "Ngôn ngữ/Loại (cho quy đổi LOC-FP):",
    options=list(LOC_PER_FP_LANGUAGES.keys()),
    key="lang_type_conversion"
)

# Các trường nhập LOC, FP, UCP - giá trị sẽ được cập nhật dựa trên primary_metric_source
# Khởi tạo giá trị trong session_state nếu chưa có để tránh lỗi khi truy cập lần đầu
if 'loc_input_val' not in st.session_state: st.session_state.loc_input_val = 10000
if 'fp_input_val' not in st.session_state: st.session_state.fp_input_val = 0.0  # Sẽ được tính
if 'ucp_input_val' not in st.session_state: st.session_state.ucp_input_val = 0.0  # Sẽ được tính

# Sử dụng st.session_state để lưu trữ giá trị người dùng nhập cho metric chính
# và các giá trị được tính toán cho các metric khác.
# Điều này cho phép các giá trị được giữ lại giữa các lần chạy lại khi nhấn nút.

col_size1, col_size2 = st.sidebar.columns(2)
with col_size1:
    # Chỉ cho phép nhập vào metric được chọn là chính, các metric khác sẽ hiển thị giá trị tính toán
    if st.session_state.primary_metric_source == 'LOC':
        st.session_state.loc_input_val = st.number_input("Lines of Code (LOC)", min_value=0,
                                                         value=st.session_state.loc_input_val, step=100,
                                                         key="loc_input_main")
        st.markdown(f"**FP (tính toán):** `{st.session_state.fp_input_val:.2f}`")
        st.markdown(f"**UCP (tính toán):** `{st.session_state.ucp_input_val:.2f}`")
    elif st.session_state.primary_metric_source == 'FP':
        st.session_state.fp_input_val = st.number_input("Function Points (FP)", min_value=0.0,
                                                        value=st.session_state.fp_input_val, step=1.0, format="%.2f",
                                                        key="fp_input_main")
        st.markdown(f"**LOC (tính toán):** `{st.session_state.loc_input_val:,.0f}`")
        st.markdown(f"**UCP (tính toán):** `{st.session_state.ucp_input_val:.2f}`")
    elif st.session_state.primary_metric_source == 'UCP':
        st.session_state.ucp_input_val = st.number_input("Use Case Points (UCP)", min_value=0.0,
                                                         value=st.session_state.ucp_input_val, step=1.0, format="%.2f",
                                                         key="ucp_input_main")
        st.markdown(f"**LOC (tính toán):** `{st.session_state.loc_input_val:,.0f}`")
        st.markdown(f"**FP (tính toán):** `{st.session_state.fp_input_val:.2f}`")

# Các trường hiển thị cho Thời gian phát triển và Team size (sẽ được tính toán)
with col_size2:  # Hoặc một khu vực riêng
    st.markdown("**Kết quả COCOMO II (mở rộng):**")
    st.markdown(f"T.Gian P.Triển (tháng):")
    st.markdown(
        f"<h5 style='text-align: left; color: orange;'>{st.session_state.get('cocomo_tdev_months', 'N/A')}</h5>",
        unsafe_allow_html=True)
    st.markdown(f"K.Thước Đội ngũ (người):")
    st.markdown(f"<h5 style='text-align: left; color: orange;'>{st.session_state.get('cocomo_avg_staff', 'N/A')}</h5>",
                unsafe_allow_html=True)

# Tham số cho mô hình truyền thống (COCOMO, FP Effort, UCP Effort)
st.sidebar.subheader("Tham số cho Mô hình Truyền thống")
cocomo_mode = st.sidebar.selectbox("Chế độ Dự án COCOMO", ["Organic", "Semi-detached", "Embedded"], key="cocomo_mode")
eaf = st.sidebar.number_input("Hệ số Điều chỉnh Nỗ lực COCOMO (EAF)", min_value=0.1, value=1.0, step=0.1, format="%.2f",
                              key="eaf")
hours_per_month = st.sidebar.number_input("Số giờ làm việc/tháng (quy đổi)", min_value=1, value=152, step=8,
                                          key="hrs_month")

st.sidebar.markdown("**Function Points (FP) Effort**")
hours_per_fp = st.sidebar.number_input("Số giờ/Function Point (Năng suất)", min_value=0.1, value=10.0, step=0.5,
                                       format="%.1f", key="hrs_fp")

st.sidebar.markdown("**Use Case Points (UCP) Effort**")
hours_per_ucp = st.sidebar.number_input("Số giờ/Use Case Point (Năng suất)", min_value=0.1, value=20.0, step=1.0,
                                        format="%.1f", key="hrs_ucp")

# Input cho ML (nếu có)
input_values_ml = {}  # Dictionary riêng cho ML features
if artifacts_loaded_successfully and preprocessor and original_cols_order and categorical_features_options:
    st.sidebar.subheader("Đặc trưng Bổ sung (cho ML)")
    # Ví dụ: nếu ML model của bạn có các feature này
    # Chúng ta sẽ không tự động điền 'Development Time (months)' và 'Team Size' từ COCOMO vào đây
    # trừ khi đó là một phần của thiết kế feature engineering của bạn.
    # Hiện tại, chúng ta coi chúng là output của COCOMO, không phải input cho ML.
    if 'Development Time (months)' in original_cols_order:
        input_values_ml['Development Time (months)'] = st.sidebar.number_input("ML: Development Time (months)",
                                                                               min_value=1, value=6, step=1,
                                                                               key="ml_dev_time")
    if 'Team Size' in original_cols_order:
        input_values_ml['Team Size'] = st.sidebar.number_input("ML: Team Size", min_value=1, value=5, step=1,
                                                               key="ml_team_size")

    # Các input phân loại cho ML
    if any(col in original_cols_order for col in categorical_features_options.keys()):
        st.sidebar.subheader("Thông tin Phân loại (cho ML)")
        col_cat_ml1, col_cat_ml2 = st.sidebar.columns(2)
        cat_cols_for_ml = [col for col in original_cols_order if col in categorical_features_options]

        with col_cat_ml1:
            for i, col_name in enumerate(cat_cols_for_ml):
                if i < len(cat_cols_for_ml) / 2:
                    options = categorical_features_options[col_name]
                    input_values_ml[col_name] = st.selectbox(f"ML: {col_name}", options=options, index=0,
                                                             key=f"ml_sb_{col_name}_1")
        with col_cat_ml2:
            for i, col_name in enumerate(cat_cols_for_ml):
                if i >= len(cat_cols_for_ml) / 2:
                    options = categorical_features_options[col_name]
                    input_values_ml[col_name] = st.selectbox(f"ML: {col_name}", options=options, index=0,
                                                             key=f"ml_sb_{col_name}_2")
else:
    st.sidebar.warning("ML: Không thể tải preprocessor hoặc thông tin cột. Phần nhập liệu ML bị hạn chế.")

# --- Nút Dự đoán/Tính toán ---
calculate_button = st.sidebar.button("📊 Ước tính & So sánh Effort", use_container_width=True, type="primary")

# --- Xử lý và Hiển thị Kết quả ---
if calculate_button:
    st.divider()
    st.subheader("📊 Kết quả Ước tính Tổng hợp")

    # 1. Thực hiện quy đổi kích thước dựa trên lựa chọn của người dùng
    primary_metric_type = st.session_state.primary_metric_source
    primary_value = 0
    if primary_metric_type == 'LOC':
        primary_value = st.session_state.loc_input_val
    elif primary_metric_type == 'FP':
        primary_value = st.session_state.fp_input_val
    elif primary_metric_type == 'UCP':
        primary_value = st.session_state.ucp_input_val

    selected_loc_per_fp = get_loc_per_fp(st.session_state.lang_type_conversion)

    converted_sizes = convert_software_sizes(
        primary_value,
        primary_metric_type,
        selected_loc_per_fp,
        UCP_TO_FP_RATIO,
        FP_TO_UCP_RATIO
    )

    if converted_sizes['error']:
        st.error(f"Lỗi quy đổi kích thước: {converted_sizes['error']}")
        st.stop()

    # Cập nhật session_state với các giá trị đã quy đổi để hiển thị lại trên sidebar
    st.session_state.loc_input_val = converted_sizes['loc']
    st.session_state.fp_input_val = converted_sizes['fp']
    st.session_state.ucp_input_val = converted_sizes['ucp']

    # Gán giá trị đã quy đổi để sử dụng trong các tính toán tiếp theo
    current_loc = converted_sizes['loc']
    current_fp = converted_sizes['fp']
    current_ucp = converted_sizes['ucp']

    st.markdown(f"**Thông tin Kích thước đã Quy đổi (sử dụng cho các ước tính):**")
    st.markdown(f"- **LOC:** `{current_loc:,.0f}`")
    st.markdown(
        f"- **FP:** `{current_fp:.2f}` (từ {primary_metric_type}={primary_value} với Ngôn ngữ: {st.session_state.lang_type_conversion}, LOC/FP={selected_loc_per_fp})")
    st.markdown(
        f"- **UCP:** `{current_ucp:.2f}` (từ {primary_metric_type}={primary_value} với UCP/FP Ratio={UCP_TO_FP_RATIO})")

    all_results = OrderedDict()
    error_messages_ml = {}

    # --- 2. Tính toán từ Mô hình Truyền thống (sử dụng giá trị đã quy đổi) ---
    st.markdown("#### A. Ước tính từ Mô hình Truyền thống")
    traditional_captions = []

    # COCOMO II Extended (Effort, TDEV, Staff)
    cocomo_results = calculate_cocomo_ii_extended(current_loc, cocomo_mode, eaf, hours_per_month, COCOMO_PARAMS)
    if cocomo_results['error']:
        st.error(f"Lỗi COCOMO II: {cocomo_results['error']}")
        all_results['COCOMO II (Effort)'] = "Lỗi"
        st.session_state.cocomo_tdev_months = "Lỗi"
        st.session_state.cocomo_avg_staff = "Lỗi"
    else:
        all_results['COCOMO II (Effort)'] = cocomo_results['effort_hrs']
        st.session_state.cocomo_tdev_months = cocomo_results['tdev_months']
        st.session_state.cocomo_avg_staff = cocomo_results['avg_staff']
        traditional_captions.append(
            f"* **COCOMO II:** Mode={cocomo_mode}, LOC={current_loc:,.0f}, EAF={eaf}, Hrs/Month={hours_per_month} "
            f"-> Effort PM={cocomo_results['effort_pm']}, TDEV={cocomo_results['tdev_months']} tháng, Staff={cocomo_results['avg_staff']} người."
        )

    # Function Points Effort
    fp_effort = calculate_fp_effort(current_fp, hours_per_fp)
    all_results['Function Points (Effort)'] = fp_effort
    traditional_captions.append(f"* **Function Points Effort:** FP={current_fp:.2f}, Hours/FP={hours_per_fp}")

    # Use Case Points Effort
    ucp_effort = calculate_ucp_effort(current_ucp, hours_per_ucp)
    all_results['Use Case Points (Effort)'] = ucp_effort
    traditional_captions.append(f"* **Use Case Points Effort:** UCP={current_ucp:.2f}, Hours/UCP={hours_per_ucp}")

    st.markdown("**Tham số và Kết quả Chi tiết (Truyền thống):**")
    for caption in traditional_captions:
        st.markdown(caption)
    st.caption("Lưu ý: Effort được tính bằng person-hours. 'Lỗi' xuất hiện nếu đầu vào không hợp lệ.")

    # --- 3. Dự đoán từ Mô hình Machine Learning ---
    if artifacts_loaded_successfully and preprocessor and feature_names_out and ml_models and original_cols_order and categorical_features_options:
        st.markdown("#### B. Dự đoán từ Mô hình Machine Learning")

        # Chuẩn bị input_df_ml. Cần đảm bảo các cột 'LOC', 'FP', 'UCP' (nếu có trong ML model)
        # sử dụng các giá trị đã quy đổi (current_loc, current_fp, current_ucp).
        # Các giá trị khác lấy từ input_values_ml.

        ml_input_data_prepared = {}
        for col in original_cols_order:
            if col == 'LOC':
                ml_input_data_prepared[col] = current_loc
            elif col == 'FP':
                ml_input_data_prepared[col] = current_fp
            elif col == 'UCP':
                ml_input_data_prepared[col] = current_ucp
            elif col in input_values_ml:  # Các feature khác của ML
                ml_input_data_prepared[col] = input_values_ml[col]
            else:
                # Nếu cột ML không phải LOC/FP/UCP và cũng không có input riêng,
                # có thể cần giá trị mặc định hoặc báo lỗi.
                # Hiện tại, preprocessor sẽ xử lý SimpleImputer nếu thiếu.
                ml_input_data_prepared[col] = np.nan
                st.warning(f"ML Input: Thiếu giá trị cho '{col}', sẽ được imputer xử lý (nếu có).")

        try:
            input_df_ml = pd.DataFrame([ml_input_data_prepared], columns=original_cols_order)
            input_processed_np = preprocessor.transform(input_df_ml)

            if isinstance(feature_names_out, list) and len(feature_names_out) == input_processed_np.shape[1]:
                input_processed_df = pd.DataFrame(input_processed_np, columns=feature_names_out)
                for model_name, loaded_model in ml_models.items():
                    try:
                        pred = loaded_model.predict(input_processed_df)
                        prediction_value = float(pred[0]) if pred.size > 0 else 0.0
                        all_results[f"ML: {model_name}"] = max(0.0, round(prediction_value, 2))
                    except Exception as model_pred_e:
                        error_msg = f"Lỗi dự đoán {model_name}: {str(model_pred_e)}"
                        st.error(error_msg)
                        all_results[f"ML: {model_name}"] = "Lỗi"
                        error_messages_ml[model_name] = str(model_pred_e)
            else:
                st.error(
                    f"Lỗi ML: Số tên đặc trưng ({len(feature_names_out)}) không khớp số cột sau transform ({input_processed_np.shape[1]}).")
                for model_name in ml_models.keys(): all_results[f"ML: {model_name}"] = "Lỗi (Config)"
        except Exception as e_ml_process:
            st.error(f"Lỗi xử lý/dự đoán ML: {e_ml_process}")
            # traceback.print_exc() # In ra console server
            for model_name in ml_models.keys(): all_results[f"ML: {model_name}"] = "Lỗi (Process)"
    else:
        st.info("Phần dự đoán Machine Learning không thực hiện do thiếu thành phần.")

    # --- 4. Hiển thị Bảng và Biểu đồ So sánh Tổng hợp ---
    st.markdown("#### C. Bảng và Biểu đồ So sánh Effort (person-hours)")
    if all_results:
        result_list = [{'Mô Hình Ước Tính': name, 'Effort Dự đoán (person-hours)': effort} for name, effort in
                       all_results.items()]
        result_df = pd.DataFrame(result_list)


        def format_effort_display(x):
            if isinstance(x, (int, float)): return f"{x:,.2f}"
            return str(x)


        st.dataframe(
            result_df.style.format({'Effort Dự đoán (person-hours)': format_effort_display}),
            use_container_width=True, hide_index=True
        )

        st.write("Biểu đồ so sánh Effort:")
        try:
            chart_df = result_df.copy()
            chart_df['Effort Dự đoán (person-hours)'] = chart_df['Effort Dự đoán (person-hours)'].astype(
                str).str.replace(',', '', regex=False)
            chart_df['Effort Dự đoán (person-hours)'] = pd.to_numeric(chart_df['Effort Dự đoán (person-hours)'],
                                                                      errors='coerce')
            chart_df.dropna(subset=['Effort Dự đoán (person-hours)'], inplace=True)

            if not chart_df.empty:
                chart_data = chart_df.set_index('Mô Hình Ước Tính')['Effort Dự đoán (person-hours)']
                st.bar_chart(chart_data)
            else:
                st.info("Không có dự đoán effort hợp lệ để vẽ biểu đồ.")
        except Exception as chart_e:
            st.warning(f"Không thể vẽ biểu đồ: {chart_e}")
            # traceback.print_exc()

    if error_messages_ml:
        st.subheader("⚠️ Chi tiết lỗi dự đoán ML:")
        for model_name, msg in error_messages_ml.items():
            st.caption(f"**{model_name}:** {msg}")

    st.info("""
    **Lưu ý quan trọng:**
    * Kết quả từ các mô hình chỉ là **ước tính**. Effort thực tế có thể khác biệt.
    * Độ chính xác phụ thuộc vào chất lượng dữ liệu huấn luyện (ML) và lựa chọn tham số (truyền thống).
    * Hãy sử dụng kết quả này như một điểm tham khảo.
    """)
    # Force a rerun to update sidebar display with new calculated values
    st.experimental_rerun()

# --- Xử lý trường hợp không tải được artifacts ban đầu ---
if not calculate_button and not artifacts_loaded_successfully:  # Chỉ hiển thị nếu chưa nhấn nút và có lỗi load
    if not ml_models and not preprocessor:
        st.error("Không thể tải các thành phần cho dự đoán Machine Learning.")
    elif not ml_models:
        st.warning("Không tải được mô hình ML. Chỉ có thể dùng mô hình truyền thống.")
    elif not preprocessor or not feature_names_out or not original_cols_order or not categorical_features_options:
        st.error("Không thể tải preprocessor/thông tin đặc trưng cho ML.")

# --- Chân trang ---
st.markdown("---")
st.caption("Ứng dụng demo được xây dựng với Streamlit, Scikit-learn, XGBoost và các mô hình ước tính truyền thống.")
