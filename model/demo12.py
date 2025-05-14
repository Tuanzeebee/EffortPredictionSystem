# Import các thư viện cần thiết
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import traceback

# --- Cấu hình trang Streamlit (PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN) ---
st.set_page_config(layout="wide", page_title="Ước tính Effort PM")

# --- Khởi tạo cờ chống vòng lặp ---
if 'processing_language_change' not in st.session_state:
    st.session_state.processing_language_change = False

# --- Hằng số và Dữ liệu Mô phỏng ---
COCOMO_A = 2.4  # Thường dùng cho Organic mode, a
COCOMO_B = 1.05  # Thường dùng cho Organic mode, b
COCOMO_C = 2.5  # Thường dùng cho Organic mode, c (cho Development Time)
COCOMO_D = 0.38  # Thường dùng cho Organic mode, d (cho Development Time)

COCOMO_II_PARAMS_BY_MODE = {
    "Organic": {"a": 2.4, "b": 1.05, "c": 2.5, "d": 0.38},
    "Semi-detached": {"a": 3.0, "b": 1.12, "c": 2.5, "d": 0.35},
    "Embedded": {"a": 3.6, "b": 1.20, "c": 2.5, "d": 0.32}
}
EFFORT_PER_UCP = 20
HOURS_PER_PERSON_MONTH = 152
AVG_LOC_PER_FP = {
    'Java': 53, 'Python': 35, 'C++': 47, 'C#': 54, 'JavaScript': 47,
    'SQL': 15, 'COBOL': 90, 'ABAP': 70, 'PHP': 40, 'Swift': 30,
    'Kotlin': 32, 'Ruby': 25, 'Go': 45, 'Assembly': 200,
    'Scripting': 20, 'Visual Basic': 32, 'Ada': 71, 'Perl': 27,
    'Khác': 50
}
LANGUAGE_TO_GL_MAP = {
    'Java': '3GL', 'Python': '3GL', 'C++': '3GL', 'C#': '3GL',
    'JavaScript': 'Scripting',
    'SQL': 'Ngôn ngữ truy vấn (SQL)',
    'COBOL': '3GL',
    'ABAP': '4GL',
    'PHP': 'Scripting',
    'Swift': '3GL', 'Kotlin': '3GL', 'Ruby': 'Scripting', 'Go': '3GL',
    'Assembly': 'Assembly',
    'Scripting': 'Scripting',
    'Visual Basic': '3GL',
    'Ada': '3GL',
    'Perl': 'Scripting',
    'Khác': 'Khác'
}

# --- Định nghĩa đường dẫn ---
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

# --- Khởi tạo biến cấu hình ---
ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = []
NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
FEATURE_NAMES_AFTER_PROCESSING = []
CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = {}

PROJECT_TYPES_OPTIONS_UI = ['Phát triển mới', 'Nâng cấp lớn', 'Khác']
LANGUAGE_TYPES_OPTIONS_UI = ['3GL', '4GL', 'Scripting', 'Ngôn ngữ truy vấn (SQL)', 'Assembly', 'Khác']
PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(list(AVG_LOC_PER_FP.keys()))
COUNT_APPROACH_OPTIONS_UI = ['IFPUG', 'NESMA', 'FiSMA', 'COSMIC', 'Mark II', 'LOC Manual', 'Khác']
APPLICATION_GROUP_OPTIONS_UI = ['Nghiệp vụ (Business)', 'Hỗ trợ Quyết định (Decision Support)',
                                'Khoa học/Kỹ thuật (Scientific/Engineering)', 'Thời gian thực (Real-time)',
                                'Hệ thống (System Software)', 'Tiện ích (Utility)', 'Khác']
APPLICATION_TYPES_OPTIONS_UI = ['Ứng dụng Web', 'Ứng dụng Di động', 'Ứng dụng Desktop', 'Hệ thống Nhúng',
                                'Xử lý Dữ liệu/Batch', 'API/Dịch vụ', 'Trí tuệ nhân tạo/ML', 'Game', 'Khác']
DEVELOPMENT_TYPES_OPTIONS_UI = ['Nội bộ (In-house)', 'Thuê ngoài (Outsource)', 'Hỗn hợp (Hybrid)',
                                'Mã nguồn mở (Đóng góp)', 'Sản phẩm (COTS) tùy chỉnh', 'Khác']


# --- Định nghĩa callbacks ---
def sync_languages_from_conversion():
    if st.session_state.processing_language_change:
        return
    st.session_state.processing_language_change = True

    new_conversion_lang = st.session_state.get('lang_conversion_v6')
    if new_conversion_lang:
        # Cập nhật PPL (ML)
        ppl_ml_key = "ml_cat_Primary Programming Language"
        if new_conversion_lang in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI:
            st.session_state[ppl_ml_key] = new_conversion_lang
        # else: # Xử lý trường hợp new_conversion_lang không có trong options PPL (ML) (hiếm khi xảy ra nếu dùng chung list)
        # st.session_state[ppl_ml_key] = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI[0] if PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI else None

        # Cập nhật Language Type (ML)
        lang_type_ml_key = "ml_cat_Language Type"
        lang_type_val = LANGUAGE_TO_GL_MAP.get(new_conversion_lang, 'Khác')
        valid_lang_type_options = LANGUAGE_TYPES_OPTIONS_UI
        if lang_type_val not in valid_lang_type_options and valid_lang_type_options:
            lang_type_val = valid_lang_type_options[0] if valid_lang_type_options[0] != "Lỗi: Ko có options" else 'Khác'
        st.session_state[lang_type_ml_key] = lang_type_val

    st.session_state.processing_language_change = False


def sync_languages_from_ppl_ml():
    if st.session_state.processing_language_change:
        return
    st.session_state.processing_language_change = True

    new_ppl_ml_lang = st.session_state.get("ml_cat_Primary Programming Language")
    if new_ppl_ml_lang:
        # Cập nhật Ngôn ngữ (cho quy đổi)
        if new_ppl_ml_lang in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI:
            st.session_state.lang_conversion_v6 = new_ppl_ml_lang

        # Cập nhật Language Type (ML)
        lang_type_ml_key = "ml_cat_Language Type"
        lang_type_val = LANGUAGE_TO_GL_MAP.get(new_ppl_ml_lang, 'Khác')
        valid_lang_type_options = LANGUAGE_TYPES_OPTIONS_UI
        if lang_type_val not in valid_lang_type_options and valid_lang_type_options:
            lang_type_val = valid_lang_type_options[0] if valid_lang_type_options[0] != "Lỗi: Ko có options" else 'Khác'
        st.session_state[lang_type_ml_key] = lang_type_val

    st.session_state.processing_language_change = False


# --- Hàm Tính Toán ---
def calculate_metrics(size_metric_choice, size_metric_value, language):
    calculated_loc = 0.0
    calculated_fp = 0.0
    calculated_ucp = 0.0
    estimated_effort_pm = 0.0
    estimated_dev_time_months = 0.0
    estimated_team_size = 0.0
    loc_fp_ratio = AVG_LOC_PER_FP.get(language, AVG_LOC_PER_FP['Khác'])

    if size_metric_value <= 0:
        return calculated_loc, calculated_fp, calculated_ucp, estimated_effort_pm, estimated_dev_time_months, estimated_team_size

    if size_metric_choice == 'LOC':
        calculated_loc = size_metric_value
    elif size_metric_choice == 'FP':
        calculated_fp = size_metric_value
        calculated_loc = calculated_fp * loc_fp_ratio if loc_fp_ratio > 0 else 0
    elif size_metric_choice == 'UCP':
        calculated_ucp = size_metric_value
        if EFFORT_PER_UCP > 0 and HOURS_PER_PERSON_MONTH > 0:
            effort_ph_from_ucp = calculated_ucp * EFFORT_PER_UCP
            effort_pm_from_ucp = effort_ph_from_ucp / HOURS_PER_PERSON_MONTH
            if COCOMO_II_PARAMS_BY_MODE["Organic"]["a"] > 0 and COCOMO_II_PARAMS_BY_MODE["Organic"][
                "b"] != 0 and effort_pm_from_ucp > 0:
                base_cocomo_val = effort_pm_from_ucp / COCOMO_II_PARAMS_BY_MODE["Organic"]["a"]
                if base_cocomo_val > 0:
                    kloc_from_ucp_effort = base_cocomo_val ** (1 / COCOMO_II_PARAMS_BY_MODE["Organic"]["b"])
                    calculated_loc = kloc_from_ucp_effort * 1000
                    if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio
            else:
                calculated_loc = 0
                calculated_fp = 0

    if size_metric_choice != 'UCP':
        if size_metric_choice == 'LOC':
            if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio
        _kloc_for_ucp_calc = calculated_loc / 1000
        if _kloc_for_ucp_calc > 0:
            _effort_pm_for_ucp_calc = COCOMO_II_PARAMS_BY_MODE["Organic"]["a"] * (
                        _kloc_for_ucp_calc ** COCOMO_II_PARAMS_BY_MODE["Organic"]["b"])
            if EFFORT_PER_UCP > 0:
                calculated_ucp = (_effort_pm_for_ucp_calc * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
            else:
                calculated_ucp = 0
        else:
            calculated_ucp = 0
    elif size_metric_choice == 'UCP' and calculated_loc == 0:
        pass

    final_kloc = calculated_loc / 1000
    params_basic = COCOMO_II_PARAMS_BY_MODE["Organic"]
    if final_kloc > 0:
        estimated_effort_pm = params_basic["a"] * (final_kloc ** params_basic["b"])
        if estimated_effort_pm > 0:
            estimated_dev_time_months = params_basic["c"] * (estimated_effort_pm ** params_basic["d"])
            if estimated_dev_time_months > 0:
                estimated_team_size = estimated_effort_pm / estimated_dev_time_months
            else:
                estimated_team_size = 1 if estimated_effort_pm > 0 else 0
    else:
        estimated_effort_pm = 0
        estimated_dev_time_months = 0
        estimated_team_size = 0

    return (
        round(calculated_loc, 2), round(calculated_fp, 2), round(calculated_ucp, 2),
        round(estimated_effort_pm, 2), round(estimated_dev_time_months, 2), round(estimated_team_size, 2)
    )


def estimate_cocomo_ii_full(kloc, project_type_cocomo="Organic", effort_multipliers_product=1.0):
    if kloc <= 0:
        return 0.0, 0.0, 0.0
    params = COCOMO_II_PARAMS_BY_MODE.get(project_type_cocomo, COCOMO_II_PARAMS_BY_MODE["Organic"])
    a, b, c_mode, d_mode = params["a"], params["b"], params["c"], params["d"]
    effort_pm = a * (kloc ** b) * effort_multipliers_product
    dev_time_months = 0
    if effort_pm > 0:
        dev_time_months = c_mode * (effort_pm ** d_mode)
    team_size = 0
    if dev_time_months > 0:
        team_size = effort_pm / dev_time_months
    elif effort_pm > 0:
        team_size = 1
    return round(effort_pm, 2), round(dev_time_months, 2), round(team_size, 1)


def calculate_dev_time_team_from_effort_ph(effort_ph, cocomo_c_const, cocomo_d_const, hrs_per_month_const):
    if effort_ph <= 0 or hrs_per_month_const <= 0:
        return 0.0, 0.0
    effort_pm = effort_ph / hrs_per_month_const
    dev_time_months = 0
    team_size = 0
    if effort_pm > 0:
        dev_time_months = cocomo_c_const * (effort_pm ** cocomo_d_const)
        if dev_time_months > 0:
            team_size = effort_pm / dev_time_months
        else:
            team_size = 1 if effort_pm > 0 else 0
    return round(dev_time_months, 2), round(team_size, 1)


@st.cache_resource
def load_artifacts_and_extract_config():
    loaded_preprocessor = None
    loaded_feature_names_after_processing = []
    loaded_ml_models = OrderedDict()
    extracted_original_cols_order = []
    extracted_numerical_features_raw = []
    extracted_categorical_features_raw = []
    extracted_categorical_options = {}
    all_loaded_successfully = True

    if not os.path.exists(PREPROCESSOR_PATH):
        all_loaded_successfully = False
    else:
        try:
            loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
            try:
                num_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'num')
                cat_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'cat')
                extracted_numerical_features_raw = list(num_transformer_tuple[2])
                extracted_categorical_features_raw = list(cat_transformer_tuple[2])
                extracted_original_cols_order = extracted_numerical_features_raw + extracted_categorical_features_raw
                cat_pipeline = loaded_preprocessor.named_transformers_['cat']
                onehot_encoder = cat_pipeline.named_steps['onehot']
                if hasattr(onehot_encoder, 'categories_') and len(onehot_encoder.categories_) == len(
                        extracted_categorical_features_raw):
                    for i, feature_name in enumerate(extracted_categorical_features_raw):
                        categories = onehot_encoder.categories_[i].tolist()
                        extracted_categorical_options[feature_name] = categories
                else:
                    all_loaded_successfully = False
            except Exception:
                all_loaded_successfully = False
        except Exception:
            all_loaded_successfully = False

    if not os.path.exists(FEATURES_PATH):
        all_loaded_successfully = False
    else:
        try:
            loaded_feature_names_after_processing = joblib.load(FEATURES_PATH)
            if isinstance(loaded_feature_names_after_processing, np.ndarray):
                loaded_feature_names_after_processing = loaded_feature_names_after_processing.tolist()
            if not isinstance(loaded_feature_names_after_processing, list):
                loaded_feature_names_after_processing = list(loaded_feature_names_after_processing)
        except Exception:
            all_loaded_successfully = False

    models_actually_loaded = 0
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            loaded_ml_models[name] = None
            continue
        try:
            model = joblib.load(path)
            loaded_ml_models[name] = model
            models_actually_loaded += 1
        except Exception:
            loaded_ml_models[name] = None
    if models_actually_loaded == 0:
        all_loaded_successfully = False
    return (
        loaded_preprocessor, loaded_feature_names_after_processing, loaded_ml_models,
        extracted_original_cols_order, extracted_numerical_features_raw,
        extracted_categorical_features_raw, extracted_categorical_options,
        all_loaded_successfully
    )


(preprocessor_loaded_global,
 feature_names_loaded_global,
 ml_models_loaded_global,
 original_cols_order_global,
 numerical_features_raw_global,
 categorical_features_raw_global,
 categorical_options_global,
 load_successful_global
 ) = load_artifacts_and_extract_config()

if load_successful_global and categorical_options_global:
    PROJECT_TYPES_OPTIONS_UI = categorical_options_global.get('Project Type', PROJECT_TYPES_OPTIONS_UI)
    all_gl_types_from_map = list(set(LANGUAGE_TO_GL_MAP.values()))
    current_lang_type_opts_from_preprocessor = categorical_options_global.get('Language Type', [])
    base_options_lt = ['3GL', '4GL', 'Scripting', 'Ngôn ngữ truy vấn (SQL)', 'Assembly',
                       'Khác']  # Danh sách cơ sở ban đầu
    combined_lang_types = list(
        dict.fromkeys(current_lang_type_opts_from_preprocessor + all_gl_types_from_map + base_options_lt))
    if 'Khác' in combined_lang_types:
        combined_lang_types.remove('Khác')
        LANGUAGE_TYPES_OPTIONS_UI = sorted(combined_lang_types) + ['Khác']
    else:
        LANGUAGE_TYPES_OPTIONS_UI = sorted(combined_lang_types)

    # Cập nhật PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI nếu preprocessor có định nghĩa
    ppl_opts_from_preprocessor = categorical_options_global.get('Primary Programming Language', [])
    if ppl_opts_from_preprocessor:  # Chỉ cập nhật nếu preprocessor có định nghĩa cụ thể
        PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(
            list(set(ppl_opts_from_preprocessor + PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI)))
    # else: giữ nguyên PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI từ AVG_LOC_PER_FP

    COUNT_APPROACH_OPTIONS_UI = categorical_options_global.get('Count Approach', COUNT_APPROACH_OPTIONS_UI)
    APPLICATION_GROUP_OPTIONS_UI = categorical_options_global.get('Application Group', APPLICATION_GROUP_OPTIONS_UI)
    APPLICATION_TYPES_OPTIONS_UI = categorical_options_global.get('Application Type', APPLICATION_TYPES_OPTIONS_UI)
    DEVELOPMENT_TYPES_OPTIONS_UI = categorical_options_global.get('Development Type', DEVELOPMENT_TYPES_OPTIONS_UI)

    ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = original_cols_order_global
    NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = numerical_features_raw_global
    CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = categorical_features_raw_global
    FEATURE_NAMES_AFTER_PROCESSING = feature_names_loaded_global

st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v6")

if 'results_summary_df' not in st.session_state: st.session_state.results_summary_df = None
if 'processed_input_df_display' not in st.session_state: st.session_state.processed_input_df_display = None
if 'raw_input_df_display' not in st.session_state: st.session_state.raw_input_df_display = None

with st.sidebar:
    st.header("📊 Nhập Thông tin Dự án")
    st.markdown("---")

    size_metric_choice = st.selectbox(
        "Chỉ số kích thước đầu vào:", ('LOC', 'FP', 'UCP'), key='size_metric_choice_v6'
    )
    default_val, step_val = (10000.0, 1000.0) if size_metric_choice == 'LOC' else (
    200.0, 10.0) if size_metric_choice == 'FP' else (100.0, 5.0)
    size_metric_value = st.number_input(
        f"Nhập giá trị {size_metric_choice}:", min_value=0.0, value=default_val, step=step_val,
        key='size_metric_value_v6', format="%.2f"
    )

    # Khởi tạo giá trị ban đầu cho lang_conversion_v6 nếu chưa có
    initial_lang_idx = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI.index(
        'Java') if 'Java' in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI else 0
    if 'lang_conversion_v6' not in st.session_state:
        st.session_state.lang_conversion_v6 = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI[initial_lang_idx]
        # Gọi callback đồng bộ ban đầu nếu các widget ML đã sẵn sàng (có thể cần kiểm tra)
        # Tuy nhiên, logic khởi tạo widget ML sẽ xử lý việc này.

    try:
        current_lang_conversion_idx = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI.index(
            st.session_state.lang_conversion_v6)
    except ValueError:  # Nếu giá trị trong session_state không còn hợp lệ (ví dụ options thay đổi)
        st.session_state.lang_conversion_v6 = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI[initial_lang_idx]
        current_lang_conversion_idx = initial_lang_idx

    st.selectbox(
        "Ngôn ngữ (cho quy đổi LOC/FP/UCP):", PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI,
        index=current_lang_conversion_idx,
        key='lang_conversion_v6',
        on_change=sync_languages_from_conversion,
        help="Chọn ngôn ngữ chính của dự án để hỗ trợ quy đổi giữa LOC, FP, UCP."
    )
    selected_primary_lang_for_conversion_val = st.session_state.lang_conversion_v6

    (calc_loc, calc_fp, calc_ucp, est_effort_pm_basic, est_dev_time_basic, est_team_size_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, selected_primary_lang_for_conversion_val
    )

    st.markdown("---")
    st.subheader("📈 Kích thước Ước tính:")
    col_loc, col_fp, col_ucp_sb = st.columns(3)
    col_loc.metric("LOC", f"{calc_loc:,.0f}", "Đầu vào" if size_metric_choice == 'LOC' else "T.Toán", delta_color="off")
    col_fp.metric("FP", f"{calc_fp:,.0f}", "Đầu vào" if size_metric_choice == 'FP' else "T.Toán", delta_color="off")
    col_ucp_sb.metric("UCP", f"{calc_ucp:,.0f}", "Đầu vào" if size_metric_choice == 'UCP' else "T.Toán",
                      delta_color="off")

    st.markdown("---")
    st.subheader("⏱️ COCOMO Cơ bản Ước tính:")
    col_e_pm_sb, col_t_m_sb, col_s_p_sb = st.columns(3)
    col_e_pm_sb.metric("Effort (PM)", f"{est_effort_pm_basic:,.1f}")
    col_t_m_sb.metric("T.Gian P.T (Tháng)", f"{est_dev_time_basic:,.1f}")
    col_s_p_sb.metric("Quy mô Nhóm", f"{est_team_size_basic:,.1f}")

    st.markdown("---")
    st.subheader("📋 Thông tin Chi Tiết cho Model ML:")

    if load_successful_global and preprocessor_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        auto_filled_values = {
            'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
            'Development Time (months)': est_dev_time_basic,
            'Team Size': est_team_size_basic
        }

        for feature_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            if feature_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                num_key = f"ml_num_{feature_name}"
                if feature_name in auto_filled_values:
                    # Không tạo widget, giá trị sẽ được lấy từ auto_filled_values khi submit
                    pass
                else:
                    if num_key not in st.session_state:
                        st.session_state[num_key] = 0.0
                    st.number_input(
                        f"{feature_name} (ML):", value=st.session_state[num_key], format="%.2f", key=num_key
                    )
            elif feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                widget_key = f"ml_cat_{feature_name}"

                # Khởi tạo giá trị cho các widget ML nếu chưa có (cho lần chạy đầu hoặc sau khi clear session)
                if widget_key not in st.session_state:
                    initial_cat_val = None
                    if feature_name == 'Primary Programming Language':
                        initial_cat_val = selected_primary_lang_for_conversion_val
                    elif feature_name == 'Language Type':
                        # Dựa trên PPL (ML) nếu đã có, nếu không thì dựa trên lang_conversion
                        ppl_for_lt_init = st.session_state.get("ml_cat_Primary Programming Language",
                                                               selected_primary_lang_for_conversion_val)
                        initial_cat_val = LANGUAGE_TO_GL_MAP.get(ppl_for_lt_init, 'Khác')
                    else:  # Các categorical khác, lấy giá trị đầu tiên từ options của preprocessor
                        options_temp = categorical_options_global.get(feature_name, [])
                        if options_temp: initial_cat_val = options_temp[0]

                    if initial_cat_val: st.session_state[widget_key] = initial_cat_val
                    # Nếu không có initial_val, Streamlit sẽ tự chọn index 0 nếu options có

                options_from_preprocessor = categorical_options_global.get(feature_name, [])
                final_options_for_selectbox = options_from_preprocessor if options_from_preprocessor else [
                    "Lỗi: Ko có options"]
                help_text_for_selectbox = None
                on_change_callback = None

                if feature_name == 'Language Type':
                    final_options_for_selectbox = LANGUAGE_TYPES_OPTIONS_UI
                    current_ppl_ml_for_help = st.session_state.get("ml_cat_Primary Programming Language",
                                                                   selected_primary_lang_for_conversion_val)
                    suggested_lt_for_help = LANGUAGE_TO_GL_MAP.get(current_ppl_ml_for_help, 'Khác')
                    help_text_for_selectbox = f"Gợi ý: {suggested_lt_for_help} (từ Ngôn ngữ ML: {current_ppl_ml_for_help})"

                elif feature_name == 'Primary Programming Language':
                    final_options_for_selectbox = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI
                    on_change_callback = sync_languages_from_ppl_ml

                current_selection_val_for_widget = st.session_state.get(widget_key)

                # Đảm bảo current_selection_val_for_widget hợp lệ trong final_options_for_selectbox
                if not current_selection_val_for_widget or current_selection_val_for_widget not in final_options_for_selectbox:
                    # Nếu không hợp lệ, thử fallback dựa trên logic đồng bộ hoặc chọn cái đầu tiên
                    if feature_name == 'Primary Programming Language':
                        current_selection_val_for_widget = st.session_state.get('lang_conversion_v6',
                                                                                final_options_for_selectbox[
                                                                                    0] if final_options_for_selectbox else None)
                    elif feature_name == 'Language Type':
                        ppl_val = st.session_state.get("ml_cat_Primary Programming Language",
                                                       st.session_state.get('lang_conversion_v6'))
                        current_selection_val_for_widget = LANGUAGE_TO_GL_MAP.get(ppl_val, 'Khác')

                    # Fallback cuối cùng nếu vẫn không có hoặc không hợp lệ
                    if not current_selection_val_for_widget or current_selection_val_for_widget not in final_options_for_selectbox:
                        if final_options_for_selectbox and final_options_for_selectbox[0] != "Lỗi: Ko có options":
                            current_selection_val_for_widget = final_options_for_selectbox[0]
                        elif final_options_for_selectbox:  # trường hợp chỉ có "Lỗi: Ko có options"
                            current_selection_val_for_widget = final_options_for_selectbox[0]

                    st.session_state[widget_key] = current_selection_val_for_widget  # Cập nhật session_state

                idx_cat = 0
                if current_selection_val_for_widget and final_options_for_selectbox and current_selection_val_for_widget in final_options_for_selectbox:
                    try:
                        idx_cat = final_options_for_selectbox.index(current_selection_val_for_widget)
                    except ValueError:
                        idx_cat = 0  # Nên không xảy ra nếu kiểm tra "in" ở trên đúng
                elif final_options_for_selectbox:
                    idx_cat = 0

                st.selectbox(
                    f"{feature_name} (ML):",
                    options=final_options_for_selectbox,
                    index=idx_cat,
                    key=widget_key,
                    on_change=on_change_callback,
                    help=help_text_for_selectbox
                )
    else:
        st.warning("Lỗi tải tài nguyên ML. Không thể tạo các trường nhập liệu chi tiết.")

    st.markdown("---")
    predict_disabled = not load_successful_global
    if st.button("🚀 Ước tính Nỗ lực Tổng hợp", key='predict_btn_v6', disabled=predict_disabled):
        final_input_dict_for_ml = {}
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            for col_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
                if col_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                    if col_name in auto_filled_values:
                        final_input_dict_for_ml[col_name] = auto_filled_values[col_name]
                    else:
                        final_input_dict_for_ml[col_name] = st.session_state.get(f"ml_num_{col_name}", 0.0)
                elif col_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                    final_input_dict_for_ml[col_name] = st.session_state.get(f"ml_cat_{col_name}")

        input_df_raw_ml = pd.DataFrame([final_input_dict_for_ml])
        try:
            if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
                input_df_raw_ml = input_df_raw_ml[ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR]
            st.session_state.raw_input_df_display = input_df_raw_ml.copy()
        except KeyError as e:
            st.error(f"Lỗi sắp xếp cột cho preprocessor: {e}.")
            st.stop()
        except Exception as e_general:
            st.error(f"Lỗi DataFrame đầu vào thô: {e_general}")
            st.stop()

        processed_df_for_model = pd.DataFrame()
        ml_processing_ok = False
        if preprocessor_loaded_global and not input_df_raw_ml.empty:
            try:
                # Kiểm tra và xử lý giá trị None hoặc NaN trước khi transform
                for col in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                    if col in input_df_raw_ml.columns and input_df_raw_ml[col].isnull().any():
                        # Tìm một giá trị hợp lệ từ options của feature đó để fillna
                        # hoặc một giá trị mặc định như 'Khác' nếu có trong options
                        options_for_col = categorical_options_global.get(col, [])
                        fallback_val = 'Khác' if 'Khác' in options_for_col else (
                            options_for_col[0] if options_for_col else "N/A")
                        input_df_raw_ml[col] = input_df_raw_ml[col].fillna(fallback_val)

                input_processed_np_array = preprocessor_loaded_global.transform(input_df_raw_ml)

                if FEATURE_NAMES_AFTER_PROCESSING and len(FEATURE_NAMES_AFTER_PROCESSING) == \
                        input_processed_np_array.shape[1]:
                    processed_df_for_model = pd.DataFrame(input_processed_np_array,
                                                          columns=FEATURE_NAMES_AFTER_PROCESSING)
                    st.session_state.processed_input_df_display = processed_df_for_model.copy()
                    ml_processing_ok = True
                else:
                    st.error(
                        f"Lỗi ML: Số tên đặc trưng ({len(FEATURE_NAMES_AFTER_PROCESSING)}) không khớp ({input_processed_np_array.shape[1]}).")
            except Exception as e_proc:
                st.error(f"Lỗi áp dụng preprocessor: {e_proc}")
                st.error(f"Dữ liệu đầu vào thô cho preprocessor:\n{input_df_raw_ml.to_dict()}")
                st.error(traceback.format_exc())
        else:
            st.warning("Preprocessor/dữ liệu ML trống.")

        results_list = []
        if ml_processing_ok and not processed_df_for_model.empty and ml_models_loaded_global:
            for model_name, model_obj in ml_models_loaded_global.items():
                effort_ph_ml = "Lỗi"
                dev_time_ml = "Lỗi"
                team_size_ml = "Lỗi"
                if model_obj:
                    try:
                        pred_ph = model_obj.predict(processed_df_for_model)
                        effort_ph_ml = round(float(pred_ph[0]), 0)
                        dev_time_ml, team_size_ml = calculate_dev_time_team_from_effort_ph(
                            effort_ph_ml, COCOMO_C, COCOMO_D, HOURS_PER_PERSON_MONTH
                        )
                    except Exception:
                        effort_ph_ml = "Lỗi dự đoán"
                else:
                    effort_ph_ml = "Mô hình chưa tải"
                results_list.append({
                    'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
                    'Model Name': model_name,
                    'Effort (Person-Hours)': effort_ph_ml,
                    'Development Time (months)': dev_time_ml,
                    'Team Size': team_size_ml
                })
        else:
            if ml_models_loaded_global:
                for model_name_key in ml_models_loaded_global.keys():
                    results_list.append({
                        'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
                        'Model Name': model_name_key,
                        'Effort (Person-Hours)': "Lỗi dữ liệu/xử lý",
                        'Development Time (months)': "N/A",
                        'Team Size': "N/A"
                    })

        kloc_cocomo_ii = calc_loc / 1000
        project_type_for_cocomo_map = {'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached",
                                       'Bảo trì': "Organic", 'Tái cấu trúc': "Semi-detached",
                                       'Tích hợp hệ thống': "Embedded", 'Khác': "Organic"}

        # Đọc giá trị 'Project Type' từ st.session_state để đảm bảo lấy giá trị người dùng chọn
        project_type_val_for_cocomo = st.session_state.get('ml_cat_Project Type',
                                                           'Phát triển mới')  # Fallback nếu key không tồn tại
        cocomo_mode_calc = project_type_for_cocomo_map.get(project_type_val_for_cocomo, "Organic")

        effort_pm_cocomo_ii, dev_time_cocomo_ii, team_size_cocomo_ii = estimate_cocomo_ii_full(
            kloc_cocomo_ii, project_type_cocomo=cocomo_mode_calc
        )
        effort_ph_cocomo_ii = round(effort_pm_cocomo_ii * HOURS_PER_PERSON_MONTH, 0)
        results_list.append({
            'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
            'Model Name': "COCOMO II",
            'Effort (Person-Hours)': effort_ph_cocomo_ii if effort_pm_cocomo_ii > 0 else "Lỗi tính toán",
            'Development Time (months)': dev_time_cocomo_ii if effort_pm_cocomo_ii > 0 else "N/A",
            'Team Size': team_size_cocomo_ii if effort_pm_cocomo_ii > 0 else "N/A"
        })
        st.session_state.results_summary_df = pd.DataFrame(results_list)
        st.success("Đã hoàn thành ước tính tổng hợp!")

main_area_results = st.container()
with main_area_results:
    st.header("📊 Bảng Tổng Kết Ước Tính Nỗ Lực")
    if st.session_state.get('results_summary_df') is not None and not st.session_state.results_summary_df.empty:
        st.dataframe(st.session_state.results_summary_df.style.format({
            'LOC': "{:,.0f}", 'FP': "{:,.0f}", 'UCP': "{:,.0f}",
            'Effort (Person-Hours)': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x,
            'Development Time (months)': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x,
            'Team Size': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x,
        }), use_container_width=True)

        st.subheader("📈 Biểu đồ So sánh Effort (Person-Hours)")
        df_for_chart = st.session_state.results_summary_df.copy()
        df_for_chart['Effort (Person-Hours)'] = pd.to_numeric(df_for_chart['Effort (Person-Hours)'], errors='coerce')
        df_for_chart.dropna(subset=['Effort (Person-Hours)'], inplace=True)
        df_for_chart = df_for_chart.sort_values(by='Effort (Person-Hours)', ascending=False)
        if not df_for_chart.empty:
            fig_compare, ax_compare = plt.subplots(figsize=(10, max(6, len(df_for_chart) * 0.5)))
            bars_compare = ax_compare.bar(df_for_chart['Model Name'], df_for_chart['Effort (Person-Hours)'],
                                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                                 '#e377c2', '#7f7f7f'])
            for bar_item in bars_compare:
                y_val_bar = bar_item.get_height()
                max_effort_val = df_for_chart['Effort (Person-Hours)'].max() if not df_for_chart[
                    'Effort (Person-Hours)'].empty else 1
                plt.text(bar_item.get_x() + bar_item.get_width() / 2.0, y_val_bar + 0.01 * max_effort_val,
                         f'{y_val_bar:,.0f}', ha='center', va='bottom', fontsize=9)
            ax_compare.set_ylabel('Effort Ước tính (Person-Hours)', fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            ax_compare.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_compare)
        else:
            st.info("Không có đủ dữ liệu Effort hợp lệ để vẽ biểu đồ so sánh.")
    elif st.session_state.get('results_summary_df') is not None and st.session_state.results_summary_df.empty:
        st.info("Chưa có kết quả nào để hiển thị. Vui lòng nhấn nút 'Ước tính Nỗ lực Tổng hợp'.")
    else:
        if 'results_summary_df' not in st.session_state and not load_successful_global:
            st.error("Tải tài nguyên ban đầu thất bại. Không thể thực hiện ước tính.")
        elif 'results_summary_df' not in st.session_state:
            st.info("Nhập thông tin ở thanh bên trái và nhấn '🚀 Ước tính Nỗ lực Tổng hợp' để xem kết quả.")