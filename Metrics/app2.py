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
st.set_page_config(layout="wide", page_title="Ước tính Effort PM Runtime")

# --- Hằng số và Dữ liệu Mô phỏng ---
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
    '.Net':80,'ASP.Net':80,'PowerBuilder':20,'Javascript':80
    # 'Khác': 50
}

LANGUAGE_TO_GL_MAP = {
    'Java': '3GL', 'Python': '3GL', 'C++': '3GL', 'C#': '3GL',
    'C':'3GL',
    'Javascript':'3GL',
    '.Net':'3GL','ASP.Net':'3GL',
    'SQL': '4GL',  # Specific type for SQL
    'COBOL': '3GL',
    'ABAP': '4GL',  # Specific type for ABAP
    'PHP': '3GL',
    'Swift': '3GL', 'Kotlin': '3GL', 'Go': '3GL',
    'Visual Basic': '3GL',
    'Ada': '3GL',
    'PowerBuilder':'4GL',
    'Oracle Forms':'4GL'
    # 'Assembly': 'Assembly',
    # 'Scripting': 'Scripting', # This entry can be ambiguous if 'Scripting' is also a language.
    # Assuming 'Scripting' as a type refers to languages like JS, PHP, Ruby, Perl.
    # If 'Scripting' itself is a selectable PPL, it should be mapped, e.g. 'Scripting': 'Scripting'
    # 'Perl': 'Scripting',
    # 'JavaScript': 'Scripting',
    # 'SQL': 'Ngôn ngữ truy vấn (SQL)',  # Specific type for SQL
    # 'Khác': 'Khác'  # General fallback type
}

OUTPUT_DIR = "."
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.joblib")
MODEL_PATHS = OrderedDict([
    ('Lasso Regression', os.path.join(OUTPUT_DIR, "lasso_regression_model.joblib")),
    ('Decision Tree', os.path.join(OUTPUT_DIR, "decision_tree_model.joblib")),
    ('Random Forest', os.path.join(OUTPUT_DIR, "random_forest_model.joblib")),
    ('XGBoost', os.path.join(OUTPUT_DIR, "xgboost_model.joblib")),
    ('MLP Regressor', os.path.join(OUTPUT_DIR, "mlp_regressor_model.joblib"))
])

# --- Khởi tạo biến cấu hình (cho các tùy chọn UI và thông tin từ preprocessor) ---
ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = []
NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
FEATURE_NAMES_AFTER_PROCESSING = []
CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = {}

BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(list(AVG_LOC_PER_FP.keys()))
PROJECT_TYPES_OPTIONS_UI_DEFAULT = ['Phát triển mới', 'Nâng cấp lớn', 'Khác']
PROJECT_TYPES_OPTIONS_UI = list(PROJECT_TYPES_OPTIONS_UI_DEFAULT)

LANGUAGE_TYPES_OPTIONS_UI = sorted(list(set(val for val in LANGUAGE_TO_GL_MAP.values() if val)))
if not LANGUAGE_TYPES_OPTIONS_UI: LANGUAGE_TYPES_OPTIONS_UI = ['Khác']

COUNT_APPROACH_OPTIONS_UI_DEFAULT = ['IFPUG', 'NESMA', 'FiSMA', 'COSMIC', 'Mark II', 'LOC Manual', 'Khác']
APPLICATION_GROUP_OPTIONS_UI_DEFAULT = ['Nghiệp vụ (Business)', 'Hỗ trợ Quyết định (Decision Support)', 'Khác']
APPLICATION_TYPES_OPTIONS_UI_DEFAULT = ['Ứng dụng Web', 'Ứng dụng Di động', 'Ứng dụng Desktop', 'Khác']
DEVELOPMENT_TYPES_OPTIONS_UI_DEFAULT = ['Nội bộ (In-house)', 'Thuê ngoài (Outsource)', 'Hỗn hợp (Hybrid)', 'Khác']

COUNT_APPROACH_OPTIONS_UI = list(COUNT_APPROACH_OPTIONS_UI_DEFAULT)
APPLICATION_GROUP_OPTIONS_UI = list(APPLICATION_GROUP_OPTIONS_UI_DEFAULT)
APPLICATION_TYPES_OPTIONS_UI = list(APPLICATION_TYPES_OPTIONS_UI_DEFAULT)
DEVELOPMENT_TYPES_OPTIONS_UI = list(DEVELOPMENT_TYPES_OPTIONS_UI_DEFAULT)


# --- Định nghĩa các hàm tiện ích và callback TRƯỚC KHI SỬ DỤNG ---
def get_filtered_ppl_options(lang_type, all_ppls, lang_map):
    if not lang_type or lang_type == "Tất cả": return sorted(list(set(all_ppls)))
    filt = [l for l in all_ppls if lang_map.get(l, 'Khác') == lang_type]
    if not filt:
        if lang_type == 'Khác':
            kh_ppls = [l for l in all_ppls if lang_map.get(l, 'Khác') == 'Khác']
            if kh_ppls: return sorted(list(set(kh_ppls)))
        return ["Khác"] if "Khác" in all_ppls else (all_ppls[:1] if all_ppls else ["N/A"])
    return sorted(list(set(filt)))


def on_language_type_change():
    st.session_state.selected_language_type_v8 = st.session_state.language_type_widget_key
    # Cập nhật PPL dựa trên LT
    current_available_ppls = get_filtered_ppl_options(st.session_state.selected_language_type_v8,
                                                      ALL_PPL_OPTIONS_CONFIG, LANGUAGE_TO_GL_MAP)
    if st.session_state.selected_ppl_v8 not in current_available_ppls:
        if "Java" in current_available_ppls:
            st.session_state.selected_ppl_v8 = "Java"
        elif current_available_ppls and current_available_ppls[0] != "N/A":
            st.session_state.selected_ppl_v8 = current_available_ppls[0]
        else:
            st.session_state.selected_ppl_v8 = "Khác"  # Fallback an toàn


def on_ppl_change():
    st.session_state.selected_ppl_v8 = st.session_state.ppl_widget_key
    # Cập nhật LT dựa trên PPL
    required_lt = LANGUAGE_TO_GL_MAP.get(st.session_state.selected_ppl_v8, 'Khác')
    if st.session_state.selected_ppl_v8 == 'ABAP' and '4GL' in LANGUAGE_TYPES_OPTIONS_UI: required_lt = '4GL'
    # (Có thể thêm các logic override khác cho PPL nếu cần)
    if required_lt != st.session_state.selected_language_type_v8 and required_lt in LANGUAGE_TYPES_OPTIONS_UI:
        st.session_state.selected_language_type_v8 = required_lt


# --- Hàm Tính Toán (giữ nguyên) ---
def calculate_metrics(size_metric_choice, size_metric_value, language):
    calculated_loc = 0.0;
    calculated_fp = 0.0;
    calculated_ucp = 0.0
    estimated_effort_pm = 0.0;
    estimated_dev_time_months = 0.0;
    estimated_team_size = 0.0
    loc_fp_ratio = AVG_LOC_PER_FP.get(language, AVG_LOC_PER_FP.get('Khác', 50))
    if size_metric_value <= 0: return calculated_loc, calculated_fp, calculated_ucp, estimated_effort_pm, estimated_dev_time_months, estimated_team_size
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
            params_organic = COCOMO_II_PARAMS_BY_MODE["Organic"]
            if params_organic["a"] > 0 and params_organic["b"] != 0 and effort_pm_from_ucp > 0:
                base_cocomo_val = effort_pm_from_ucp / params_organic["a"]
                if base_cocomo_val > 0:
                    kloc_from_ucp_effort = base_cocomo_val ** (1 / params_organic["b"])
                    calculated_loc = kloc_from_ucp_effort * 1000
                    if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio
    if size_metric_choice != 'UCP':
        if size_metric_choice == 'LOC' and loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio
        _kloc_for_ucp_calc = calculated_loc / 1000
        if _kloc_for_ucp_calc > 0:
            params_organic = COCOMO_II_PARAMS_BY_MODE["Organic"]
            _effort_pm_for_ucp_calc = params_organic["a"] * (_kloc_for_ucp_calc ** params_organic["b"])
            if EFFORT_PER_UCP > 0 and HOURS_PER_PERSON_MONTH > 0: calculated_ucp = (
                                                                                               _effort_pm_for_ucp_calc * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
    final_kloc = calculated_loc / 1000
    params_basic = COCOMO_II_PARAMS_BY_MODE["Organic"]
    if final_kloc > 0:
        estimated_effort_pm = params_basic["a"] * (final_kloc ** params_basic["b"])
        if estimated_effort_pm > 0:
            estimated_dev_time_months = params_basic["c"] * (estimated_effort_pm ** params_basic["d"])
            estimated_team_size = (
                        estimated_effort_pm / estimated_dev_time_months) if estimated_dev_time_months > 0 else (
                1.0 if estimated_effort_pm > 0 else 0.0)
        else:
            estimated_effort_pm = 0.0
    return (round(calculated_loc, 2), round(calculated_fp, 2), round(calculated_ucp, 2),
            round(estimated_effort_pm, 2), round(estimated_dev_time_months, 2), round(estimated_team_size, 1))


def estimate_cocomo_ii_full(kloc, project_type_cocomo="Organic", effort_multipliers_product=1.0):
    if kloc <= 0: return 0.0, 0.0, 0.0
    params = COCOMO_II_PARAMS_BY_MODE.get(project_type_cocomo, COCOMO_II_PARAMS_BY_MODE["Organic"])
    a, b, c_mode, d_mode = params["a"], params["b"], params["c"], params["d"]
    effort_pm = a * (kloc ** b) * effort_multipliers_product
    dev_time_months = 0.0;
    team_size = 0.0
    if effort_pm > 0:
        dev_time_months = c_mode * (effort_pm ** d_mode)
        team_size = (effort_pm / dev_time_months) if dev_time_months > 0 else 1.0
    else:
        effort_pm = 0.0
    return round(effort_pm, 2), round(dev_time_months, 2), round(team_size, 1)


def calculate_dev_time_team_from_effort_ph(effort_ph, cocomo_mode_for_time_calc, hrs_per_month_const):
    if effort_ph <= 0 or hrs_per_month_const <= 0: return 0.0, 0.0
    effort_pm = effort_ph / hrs_per_month_const
    params = COCOMO_II_PARAMS_BY_MODE.get(cocomo_mode_for_time_calc, COCOMO_II_PARAMS_BY_MODE["Organic"])
    current_cocomo_c, current_cocomo_d = params["c"], params["d"]
    dev_time_months = 0.0;
    team_size = 0.0
    if effort_pm > 0:
        dev_time_months = current_cocomo_c * (effort_pm ** current_cocomo_d)
        team_size = (effort_pm / dev_time_months) if dev_time_months > 0 else 1.0
    return round(dev_time_months, 2), round(team_size, 1)


# --- Tải Artifacts (giữ nguyên @st.cache_resource) ---
@st.cache_resource
def load_artifacts_and_extract_config():
    loaded_preprocessor = None;
    loaded_feature_names_after_processing = [];
    loaded_ml_models = OrderedDict()
    extracted_original_cols_order = [];
    extracted_numerical_features_raw = [];
    extracted_categorical_features_raw = []
    extracted_categorical_options = {};
    all_loaded_successfully = True
    try:
        if not os.path.exists(PREPROCESSOR_PATH): raise FileNotFoundError()
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
        if hasattr(loaded_preprocessor, 'transformers_') and isinstance(loaded_preprocessor.transformers_, list):
            num_t = next((t for t in loaded_preprocessor.transformers_ if t[0] == 'num'), None)
            cat_t = next((t for t in loaded_preprocessor.transformers_ if t[0] == 'cat'), None)
            if num_t: extracted_numerical_features_raw = list(num_t[2])
            if cat_t:
                extracted_categorical_features_raw = list(cat_t[2])
                if hasattr(loaded_preprocessor, 'named_transformers_') and \
                        'cat' in loaded_preprocessor.named_transformers_ and \
                        hasattr(loaded_preprocessor.named_transformers_['cat'], 'named_steps') and \
                        'onehot' in loaded_preprocessor.named_transformers_['cat'].named_steps:
                    onehot_enc = loaded_preprocessor.named_transformers_['cat'].named_steps['onehot']
                    if hasattr(onehot_enc, 'categories_'):
                        for i, fname in enumerate(extracted_categorical_features_raw):
                            if i < len(onehot_enc.categories_): extracted_categorical_options[fname] = \
                            onehot_enc.categories_[i].tolist()
                else:
                    for fname in extracted_categorical_features_raw: extracted_categorical_options[fname] = []
            extracted_original_cols_order = extracted_numerical_features_raw + extracted_categorical_features_raw
        else:
            all_loaded_successfully = False
    except:
        all_loaded_successfully = False
    try:
        if not os.path.exists(FEATURES_PATH): raise FileNotFoundError()
        loaded_feature_names_after_processing = joblib.load(FEATURES_PATH)
        if isinstance(loaded_feature_names_after_processing,
                      np.ndarray): loaded_feature_names_after_processing = loaded_feature_names_after_processing.tolist()
        if not isinstance(loaded_feature_names_after_processing, list): loaded_feature_names_after_processing = list(
            loaded_feature_names_after_processing)
    except:
        all_loaded_successfully = False
    models_loaded_count = 0
    for name, path in MODEL_PATHS.items():
        try:
            if not os.path.exists(path): raise FileNotFoundError()
            loaded_ml_models[name] = joblib.load(path);
            models_loaded_count += 1
        except:
            loaded_ml_models[name] = None
    if models_loaded_count == 0 and MODEL_PATHS: all_loaded_successfully = False
    return (loaded_preprocessor, loaded_feature_names_after_processing, loaded_ml_models,
            extracted_original_cols_order, extracted_numerical_features_raw,
            extracted_categorical_features_raw, extracted_categorical_options, all_loaded_successfully)


# --- Tải dữ liệu và cấu hình một lần ---
(preprocessor_loaded_global, feature_names_loaded_global, ml_models_loaded_global,
 original_cols_order_global, numerical_features_raw_global, categorical_features_raw_global,
 categorical_options_global, load_successful_global) = load_artifacts_and_extract_config()

# --- Cập nhật các tùy chọn UI dựa trên preprocessor ---
ALL_PPL_OPTIONS_CONFIG = BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI  # Khởi tạo trước
if load_successful_global and categorical_options_global:
    CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = categorical_options_global
    ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = original_cols_order_global
    NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = numerical_features_raw_global
    CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = categorical_features_raw_global
    FEATURE_NAMES_AFTER_PROCESSING = feature_names_loaded_global
    PROJECT_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Project Type',
                                                                         PROJECT_TYPES_OPTIONS_UI_DEFAULT)
    COUNT_APPROACH_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Count Approach',
                                                                          COUNT_APPROACH_OPTIONS_UI_DEFAULT)
    APPLICATION_GROUP_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Application Group',
                                                                             APPLICATION_GROUP_OPTIONS_UI_DEFAULT)
    APPLICATION_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Application Type',
                                                                             APPLICATION_TYPES_OPTIONS_UI_DEFAULT)
    DEVELOPMENT_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Development Type',
                                                                             DEVELOPMENT_TYPES_OPTIONS_UI_DEFAULT)
    # Cập nhật ALL_PPL_OPTIONS_CONFIG nếu có từ preprocessor
    if 'Primary Programming Language' in CATEGORICAL_OPTIONS_FROM_PREPROCESSOR:
        ppl_fp_config = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR['Primary Programming Language']
        if ppl_fp_config:
            ALL_PPL_OPTIONS_CONFIG = sorted(list(set(ppl_fp_config)))
            if 'Khác' not in ALL_PPL_OPTIONS_CONFIG: ALL_PPL_OPTIONS_CONFIG.append('Khác')

# --- Tiêu đề ứng dụng ---
st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v9.1 (Runtime)")

# --- Khởi tạo Session State (cho các widget động và kết quả) ---
if 'results_summary_df' not in st.session_state: st.session_state.results_summary_df = pd.DataFrame()

if 'ui_selected_project_type' not in st.session_state:
    st.session_state.ui_selected_project_type = PROJECT_TYPES_OPTIONS_UI[
        0] if PROJECT_TYPES_OPTIONS_UI else 'Phát triển mới'

if 'selected_language_type_v8' not in st.session_state:  # Đổi tên key để tránh xung đột nếu chạy song song
    st.session_state.selected_language_type_v8 = '3GL' if '3GL' in LANGUAGE_TYPES_OPTIONS_UI else (
        LANGUAGE_TYPES_OPTIONS_UI[0] if LANGUAGE_TYPES_OPTIONS_UI else 'Khác')

if 'selected_ppl_v8' not in st.session_state:
    init_ppl_opts_s = get_filtered_ppl_options(st.session_state.selected_language_type_v8, ALL_PPL_OPTIONS_CONFIG,
                                               LANGUAGE_TO_GL_MAP)
    default_ppl = "Java"
    if default_ppl not in init_ppl_opts_s:
        if init_ppl_opts_s and init_ppl_opts_s[0] != "N/A":
            default_ppl = init_ppl_opts_s[0]
        elif "Khác" in init_ppl_opts_s:
            default_ppl = "Khác"
        else:
            default_ppl = ALL_PPL_OPTIONS_CONFIG[
                0] if ALL_PPL_OPTIONS_CONFIG else "N/A"  # Hoặc một fallback an toàn hơn
    st.session_state.selected_ppl_v8 = default_ppl

# --- Khu vực Nhập liệu Chính ---
input_section = st.container()
with input_section:
    st.header("📝 Nhập Thông tin Dự án")

    col_basic_1, col_basic_2 = st.columns(2)
    with col_basic_1:
        size_metric_choice = st.selectbox("Chỉ số kích thước đầu vào:", ('LOC', 'FP', 'UCP'),
                                          key='size_metric_choice_main')
        def_val, st_val = (10000.0, 1000.0) if size_metric_choice == 'LOC' else (
        200.0, 10.0) if size_metric_choice == 'FP' else (100.0, 5.0)
        size_metric_value = st.number_input(f"Nhập giá trị {size_metric_choice}:", min_value=0.0, value=def_val,
                                            step=st_val, key='size_metric_value_main', format="%.2f")

    with col_basic_2:
        current_project_type_options_main = PROJECT_TYPES_OPTIONS_UI if PROJECT_TYPES_OPTIONS_UI else PROJECT_TYPES_OPTIONS_UI_DEFAULT
        if st.session_state.ui_selected_project_type not in current_project_type_options_main and current_project_type_options_main:
            st.session_state.ui_selected_project_type = current_project_type_options_main[0]
        elif not current_project_type_options_main:
            st.session_state.ui_selected_project_type = "Phát triển mới";
            current_project_type_options_main = ["Phát triển mới"]

        selected_project_type_from_ui = st.selectbox(
            "Loại Dự Án Chính:", options=current_project_type_options_main,
            index=current_project_type_options_main.index(
                st.session_state.ui_selected_project_type) if st.session_state.ui_selected_project_type in current_project_type_options_main else 0,
            key="ui_project_type_selector_widget_key",
            help="Ảnh hưởng đến tham số COCOMO cho COCOMO II và suy luận Thời gian/Nhóm cho ML."
        )
        st.session_state.ui_selected_project_type = selected_project_type_from_ui

    st.markdown("---")
    st.subheader("Ngôn ngữ Lập trình")
    col_lang_1, col_lang_2 = st.columns(2)
    with col_lang_1:
        lt_idx_main = LANGUAGE_TYPES_OPTIONS_UI.index(
            st.session_state.selected_language_type_v8) if st.session_state.selected_language_type_v8 in LANGUAGE_TYPES_OPTIONS_UI else 0
        st.selectbox("Loại ngôn ngữ:", options=LANGUAGE_TYPES_OPTIONS_UI, index=lt_idx_main,
                     key="language_type_widget_key", on_change=on_language_type_change)

    with col_lang_2:
        current_ppl_opts_main = get_filtered_ppl_options(st.session_state.selected_language_type_v8,
                                                         ALL_PPL_OPTIONS_CONFIG, LANGUAGE_TO_GL_MAP)
        if st.session_state.selected_ppl_v8 not in current_ppl_opts_main:  # Tự điều chỉnh nếu PPL hiện tại không hợp lệ
            if "Java" in current_ppl_opts_main:
                st.session_state.selected_ppl_v8 = "Java"
            elif current_ppl_opts_main and current_ppl_opts_main[0] != "N/A":
                st.session_state.selected_ppl_v8 = current_ppl_opts_main[0]
            else:
                st.session_state.selected_ppl_v8 = "Khác" if "Khác" in current_ppl_opts_main else (
                    ALL_PPL_OPTIONS_CONFIG[0] if ALL_PPL_OPTIONS_CONFIG else "N/A")

        ppl_idx_main = current_ppl_opts_main.index(
            st.session_state.selected_ppl_v8) if st.session_state.selected_ppl_v8 in current_ppl_opts_main else 0
        st.selectbox("Ngôn ngữ chính:", options=current_ppl_opts_main, index=ppl_idx_main, key="ppl_widget_key",
                     on_change=on_ppl_change)

    _current_selected_ppl_for_conversion_and_ml = st.session_state.selected_ppl_v8

    input_values_for_ml_dict = {}
    if load_successful_global and preprocessor_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        with st.expander("📋 Thông tin Chi Tiết cho Model ML (tùy chọn)", expanded=False):
            col_ml_1, col_ml_2 = st.columns(2)  # Bố cục 2 cột cho input ML
            current_col_ml = col_ml_1

            for i, feature_name in enumerate(ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR):
                # Chuyển cột sau mỗi N/2 features (ví dụ)
                if i >= len(ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR) / 2 and current_col_ml == col_ml_1:
                    current_col_ml = col_ml_2

                with current_col_ml:
                    if feature_name == 'Project Type':
                        input_values_for_ml_dict[feature_name] = selected_project_type_from_ui
                        st.markdown(f"**Loại Dự Án (ML):** `{selected_project_type_from_ui}`")
                        continue
                    if feature_name == 'Language Type':
                        input_values_for_ml_dict[feature_name] = st.session_state.selected_language_type_v8
                        st.markdown(f"**Loại Ngôn Ngữ (ML):** `{st.session_state.selected_language_type_v8}`")
                        continue
                    if feature_name == 'Primary Programming Language':
                        input_values_for_ml_dict[feature_name] = _current_selected_ppl_for_conversion_and_ml
                        st.markdown(f"**Ngôn Ngữ Chính (ML):** `{_current_selected_ppl_for_conversion_and_ml}`")
                        continue

                    if feature_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                        if feature_name not in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:
                            input_values_for_ml_dict[feature_name] = st.number_input(f"{feature_name} (ML):", value=0.0,
                                                                                     format="%.2f",
                                                                                     key=f"ml_num_{feature_name}_main")
                        else:
                            input_values_for_ml_dict[feature_name] = 0.0
                    elif feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                        options = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get(feature_name, [])
                        # Xác định default_val và default_idx an toàn hơn
                        default_val = options[0] if options else None
                        if default_val is None and feature_name in input_values_for_ml_dict and \
                                input_values_for_ml_dict[feature_name] in options:  # Thử lấy từ dict nếu đã có
                            default_val = input_values_for_ml_dict[feature_name]

                        default_idx = 0
                        if default_val and options:  # Chỉ tìm index nếu default_val và options hợp lệ
                            try:
                                default_idx = options.index(default_val)
                            except ValueError:
                                default_idx = 0  # Nếu default_val không có trong options, dùng index 0

                        input_values_for_ml_dict[feature_name] = st.selectbox(f"{feature_name} (ML):",
                                                                              options=options if options else ["N/A"],
                                                                              index=default_idx,
                                                                              key=f"ml_cat_{feature_name}_main")

    elif not load_successful_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        st.warning("Không thể tải cấu hình ML. Một số trường ML có thể không chính xác.")

    (calc_loc, calc_fp, calc_ucp, est_effort_pm_basic, est_dev_time_basic, est_team_size_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, _current_selected_ppl_for_conversion_and_ml)

    auto_calc_inputs_for_ml = {'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
                               'Development Time (months)': est_dev_time_basic,
                               'Team Size': est_team_size_basic}

    for k, v in auto_calc_inputs_for_ml.items():
        if k in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and k in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
            input_values_for_ml_dict[k] = v  # Gán vào dict sẽ dùng cho ML

    exp_basic_metrics = st.expander("📏 Kích thước Dự án & Ước tính COCOMO Cơ bản (Cập nhật trực tiếp)", expanded=True)
    with exp_basic_metrics:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("LOC Ước tính", f"{calc_loc:,.0f}")
        col_m2.metric("FP Ước tính", f"{calc_fp:,.0f}")
        col_m3.metric("UCP Ước tính", f"{calc_ucp:,.0f}")
        st.markdown("---")
        col_m4, col_m5, col_m6 = st.columns(3)
        col_m4.metric("Effort Cơ bản (PM)", f"{est_effort_pm_basic:,.1f}")
        col_m5.metric("T.Gian P.T Cơ bản (Tháng)", f"{est_dev_time_basic:,.1f}")
        col_m6.metric("Quy mô Nhóm Cơ bản", f"{est_team_size_basic:,.1f}")

        if load_successful_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            st.markdown("---")
            st.write("**Dữ liệu số sẽ dùng cho Model ML:**")
            disp_ml_inputs = {}
            for col_n_ml in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # Duyệt qua các feature ML mong đợi
                if col_n_ml in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:  # Chỉ hiển thị các feature số
                    val_to_disp = input_values_for_ml_dict.get(col_n_ml)  # Lấy từ dict đã chuẩn bị cho ML
                    if val_to_disp is not None: disp_ml_inputs[col_n_ml] = f"{val_to_disp:,.1f}" if isinstance(
                        val_to_disp, (float, int)) else str(val_to_disp)
            if disp_ml_inputs:
                max_cols_disp = 3
                num_items_disp = len(disp_ml_inputs)
                cols_auto_ml = st.columns(min(num_items_disp, max_cols_disp))
                idx_auto_ml = 0
                for k_auto_ml, v_auto_ml in disp_ml_inputs.items():
                    cols_auto_ml[idx_auto_ml % max_cols_disp].metric(label=f"{k_auto_ml} (cho ML)", value=v_auto_ml)
                    idx_auto_ml += 1
            # else: st.caption("Không có giá trị số nào được cấu hình cho ML.")

# --- LOGIC TÍNH TOÁN RUNTIME ---
results_list_runtime = []
st.session_state.results_summary_df = pd.DataFrame()

can_predict_ml_runtime = (load_successful_global and preprocessor_loaded_global and ml_models_loaded_global and
                          ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and FEATURE_NAMES_AFTER_PROCESSING)

project_type_for_calc_runtime = selected_project_type_from_ui  # Đã lấy từ widget ở trên
project_type_map_runtime = {'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached", 'Khác': "Organic"}
cocomo_mode_runtime = project_type_map_runtime.get(project_type_for_calc_runtime, "Organic")

final_ml_input_dict_runtime = {}  # Sẽ được xây dựng lại từ input_values_for_ml_dict

if can_predict_ml_runtime:
    # Chuẩn bị input cho ML models từ input_values_for_ml_dict đã có giá trị từ widget và auto_calc
    for col_name_rt in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        current_val_rt = input_values_for_ml_dict.get(col_name_rt)  # Lấy giá trị đã được chuẩn bị
        if col_name_rt in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
            try:
                if current_val_rt is None:
                    final_ml_input_dict_runtime[col_name_rt] = 0.0  # Fallback nếu vẫn None
                else:
                    final_ml_input_dict_runtime[col_name_rt] = float(current_val_rt)
            except (ValueError, TypeError):
                final_ml_input_dict_runtime[col_name_rt] = 0.0  # Fallback
        else:  # Categorical
            final_ml_input_dict_runtime[
                col_name_rt] = current_val_rt if current_val_rt is not None else "N/A"  # Fallback

    try:
        raw_ml_df_runtime = pd.DataFrame([final_ml_input_dict_runtime])
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            raw_ml_df_runtime = raw_ml_df_runtime[ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR]

        processed_df_model_runtime = pd.DataFrame();
        ml_processing_ok_runtime = False
        if preprocessor_loaded_global and not raw_ml_df_runtime.empty and raw_ml_df_runtime.isnull().sum().sum() == 0:  # Kiểm tra thêm NaN
            input_processed_np_array_rt = preprocessor_loaded_global.transform(raw_ml_df_runtime)
            if FEATURE_NAMES_AFTER_PROCESSING and len(FEATURE_NAMES_AFTER_PROCESSING) == \
                    input_processed_np_array_rt.shape[1]:
                processed_df_model_runtime = pd.DataFrame(input_processed_np_array_rt,
                                                          columns=FEATURE_NAMES_AFTER_PROCESSING)
                ml_processing_ok_runtime = True

        if ml_processing_ok_runtime and not processed_df_model_runtime.empty and ml_models_loaded_global:
            for model_name, model_obj in ml_models_loaded_global.items():
                eff_ph_ml, dev_t_ml, team_s_ml = "Lỗi XL", "N/A", "N/A"  # Mặc định nếu có lỗi
                if model_obj:
                    try:
                        pred_ph = model_obj.predict(processed_df_model_runtime)[0]
                        eff_ph_ml = round(float(pred_ph), 0)
                        if eff_ph_ml < 0: eff_ph_ml = 0  # Đảm bảo effort không âm
                        dev_t_ml, team_s_ml = calculate_dev_time_team_from_effort_ph(eff_ph_ml, cocomo_mode_runtime,
                                                                                     HOURS_PER_PERSON_MONTH)
                    except Exception:
                        eff_ph_ml = "Lỗi Pred"
                else:
                    eff_ph_ml = "N/A Model"
                results_list_runtime.append({'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_name,
                                             'Effort (Person-Hours)': eff_ph_ml, 'Development Time (months)': dev_t_ml,
                                             'Team Size': team_s_ml})
        elif ml_models_loaded_global:
            for model_n_key_rt in ml_models_loaded_global.keys(): results_list_runtime.append(
                {'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_n_key_rt,
                 'Effort (Person-Hours)': "Lỗi DL/XL", 'Development Time (months)': "N/A", 'Team Size': "N/A"})

    except Exception as e_ml_block_runtime:
        # st.toast(f"Lỗi trong khối ML runtime: {str(e_ml_block_runtime)[:100]}...", icon="🔥") # Thông báo ngắn
        for model_n_key_rt_err in MODEL_PATHS.keys():
            results_list_runtime.append(
                {'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_n_key_rt_err,
                 'Effort (Person-Hours)': "Lỗi ML Chung", 'Development Time (months)': "N/A", 'Team Size': "N/A"})

# Tính COCOMO II
if calc_loc > 0:  # Chỉ tính nếu có LOC
    kloc_cocomo2_rt = calc_loc / 1000
    eff_pm_coco2_rt, dev_t_coco2_rt, team_s_coco2_rt = estimate_cocomo_ii_full(kloc_cocomo2_rt,
                                                                               project_type_cocomo=cocomo_mode_runtime)
    eff_ph_coco2_rt = round(eff_pm_coco2_rt * HOURS_PER_PERSON_MONTH, 0)
    results_list_runtime.append({'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': "COCOMO II",
                                 'Effort (Person-Hours)': eff_ph_coco2_rt if eff_pm_coco2_rt > 0 else (
                                     "0" if kloc_cocomo2_rt > 0 else "N/A"),  # Sửa N/A -> 0 nếu kloc>0
                                 'Development Time (months)': dev_t_coco2_rt if eff_pm_coco2_rt > 0 else (
                                     "0" if kloc_cocomo2_rt > 0 else "N/A"),
                                 'Team Size': team_s_coco2_rt if eff_pm_coco2_rt > 0 else (
                                     "0" if kloc_cocomo2_rt > 0 else "N/A")})
elif not results_list_runtime and not can_predict_ml_runtime:  # Nếu không có kết quả ML và không có LOC cho Cocomo
    results_list_runtime.append(
        {'LOC': "N/A", 'FP': "N/A", 'UCP': "N/A", 'Model Name': "COCOMO II", 'Effort (Person-Hours)': "Thiếu LOC",
         'Development Time (months)': "N/A", 'Team Size': "N/A"})

if results_list_runtime:
    st.session_state.results_summary_df = pd.DataFrame(results_list_runtime)

# --- Khu vực Hiển thị Kết quả Chính ---
results_section = st.container()
with results_section:
    st.markdown("---")
    st.header("💡Kết Quả Ước Tính Tổng Hợp (Cập nhật trực tiếp)")
    if not st.session_state.results_summary_df.empty:
        st.dataframe(st.session_state.results_summary_df.style.format({
            'LOC': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'FP': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'UCP': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'Effort (Person-Hours)': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'Development Time (months)': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x),
            'Team Size': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x),
        }), use_container_width=True)

        df_chart_rt = st.session_state.results_summary_df.copy()
        df_chart_rt['Effort (Person-Hours)'] = pd.to_numeric(df_chart_rt['Effort (Person-Hours)'], errors='coerce')
        df_chart_rt.dropna(subset=['Effort (Person-Hours)'], inplace=True)
        df_chart_rt_valid = df_chart_rt[df_chart_rt['Effort (Person-Hours)'] > 0].sort_values(
            by='Effort (Person-Hours)', ascending=False)  # Đổi tên biến

        if not df_chart_rt_valid.empty:  # Sử dụng biến đã lọc
            st.subheader("📈 Biểu đồ So sánh Effort (Person-Hours)")
            fig_rt, ax_rt = plt.subplots(
                figsize=(10, max(6, len(df_chart_rt_valid) * 0.6)))  # Sử dụng df_chart_rt_valid
            colors_rt = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                         '#7f7f7f']  # Thêm màu
            bars_rt = ax_rt.bar(df_chart_rt_valid['Model Name'], df_chart_rt_valid['Effort (Person-Hours)'],
                                color=[colors_rt[i % len(colors_rt)] for i in range(len(df_chart_rt_valid))])
            max_eff_rt = df_chart_rt_valid['Effort (Person-Hours)'].max() if not df_chart_rt_valid[
                'Effort (Person-Hours)'].empty else 1
            for bar_rt_item in bars_rt:
                yval_rt = bar_rt_item.get_height()
                ax_rt.text(bar_rt_item.get_x() + bar_rt_item.get_width() / 2.0, yval_rt + 0.01 * max_eff_rt,
                           f'{yval_rt:,.0f}', ha='center', va='bottom', fontsize=9)
            ax_rt.set_ylabel('Effort Ước tính (Person-Hours)');
            ax_rt.set_xlabel('Mô hình Ước tính');
            ax_rt.set_title('So sánh Effort giữa các Mô hình')
            plt.xticks(rotation=45, ha="right");
            plt.yticks();
            ax_rt.grid(axis='y', linestyle='--', alpha=0.7);
            plt.tight_layout();
            st.pyplot(fig_rt)
        # else: st.caption("Không có dữ liệu Effort hợp lệ (>0) để vẽ biểu đồ.")

    else:
        if not load_successful_global:
            st.warning("Tải tài nguyên ban đầu thất bại. Kết quả có thể không đầy đủ.")
        else:
            st.info("Nhập thông tin ở trên để xem kết quả ước tính tự động cập nhật.")

# --- Thông báo tải Artifacts ---
if not load_successful_global:
    st.toast("⚠️ LƯU Ý: Không thể tải các tệp mô hình/cấu hình. Kết quả ML có thể không khả dụng.", icon="⚙️")