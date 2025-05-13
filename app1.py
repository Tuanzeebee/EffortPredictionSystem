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
    '.Net': 80, 'ASP.Net': 80, 'PowerBuilder': 20,
    'Khác': 50
}

LANGUAGE_TO_GL_MAP = {
    'Java': '3GL', 'Python': '3GL', 'C++': '3GL', 'C#': '3GL', 'C': '3GL',
    'JavaScript': '3GL', '.Net': '3GL', 'ASP.Net': '3GL', 'SQL': '4GL',
    'COBOL': '3GL', 'ABAP': '4GL', 'PHP': '3GL', 'Swift': '3GL',
    'Kotlin': '3GL', 'Go': '3GL', 'Visual Basic': '3GL', 'Ada': '3GL',
    'PowerBuilder': '4GL', 'Oracle Forms': '4GL', 'Perl': '3GL',
    'Assembly': 'Assembly', 'Scripting': 'Scripting'
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

ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = []
NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
FEATURE_NAMES_AFTER_PROCESSING = []
CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = {}

BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(list(AVG_LOC_PER_FP.keys()))
PROJECT_TYPES_OPTIONS_UI_DEFAULT = ['Phát triển mới', 'Nâng cấp lớn', 'Khác']  # Default
PROJECT_TYPES_OPTIONS_UI = list(PROJECT_TYPES_OPTIONS_UI_DEFAULT)  # Tạo bản sao để có thể cập nhật

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
            estimated_effort_pm = 0.0  # Ensure positive
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


@st.cache_resource
def load_artifacts_and_extract_config():
    # (Nội dung hàm này giữ nguyên như trước, đã khá ổn định)
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
                onehot_enc = loaded_preprocessor.named_transformers_['cat'].named_steps['onehot']
                if hasattr(onehot_enc, 'categories_'):
                    for i, fname in enumerate(extracted_categorical_features_raw):
                        if i < len(onehot_enc.categories_): extracted_categorical_options[fname] = \
                        onehot_enc.categories_[i].tolist()
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


(preprocessor_loaded_global, feature_names_loaded_global, ml_models_loaded_global,
 original_cols_order_global, numerical_features_raw_global, categorical_features_raw_global,
 categorical_options_global, load_successful_global) = load_artifacts_and_extract_config()

if load_successful_global and categorical_options_global:
    CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = categorical_options_global
    ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = original_cols_order_global
    NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = numerical_features_raw_global
    CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = categorical_features_raw_global
    FEATURE_NAMES_AFTER_PROCESSING = feature_names_loaded_global
    # Cập nhật các danh sách lựa chọn UI từ preprocessor nếu có
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

st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v8.3")

if 'results_summary_df' not in st.session_state: st.session_state.results_summary_df = pd.DataFrame()
if 'processed_input_df_display' not in st.session_state: st.session_state.processed_input_df_display = None
if 'raw_input_df_display' not in st.session_state: st.session_state.raw_input_df_display = None

ALL_PPL_OPTIONS_CONFIG = BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI
if load_successful_global and 'Primary Programming Language' in CATEGORICAL_OPTIONS_FROM_PREPROCESSOR:
    ppl_fp = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR['Primary Programming Language']  # ppl_from_preprocessor
    if ppl_fp:
        ALL_PPL_OPTIONS_CONFIG = sorted(list(set(ppl_fp)))
        if 'Khác' not in ALL_PPL_OPTIONS_CONFIG: ALL_PPL_OPTIONS_CONFIG.append('Khác')


def get_filtered_ppl_options(lang_type, all_ppls, lang_map):  # language_type, lang_to_gl_map
    if not lang_type or lang_type == "Tất cả": return sorted(list(set(all_ppls)))
    filt = [l for l in all_ppls if lang_map.get(l, 'Khác') == lang_type]  # filtered, lang
    if not filt:
        if lang_type == 'Khác':
            kh_ppls = [l for l in all_ppls if lang_map.get(l, 'Khác') == 'Khác']  # khac_ppls, lang
            if kh_ppls: return sorted(list(set(kh_ppls)))
        return ["Khác"] if "Khác" in all_ppls else (all_ppls[:1] if all_ppls else ["N/A"])
    return sorted(list(set(filt)))


if 'app_v8_lang_init_done' not in st.session_state:  # Khởi tạo session state cho ngôn ngữ
    st.session_state.selected_language_type_v8 = '3GL' if '3GL' in LANGUAGE_TYPES_OPTIONS_UI else (
        LANGUAGE_TYPES_OPTIONS_UI[0] if LANGUAGE_TYPES_OPTIONS_UI else 'Khác')
    init_ppl_opts = get_filtered_ppl_options(st.session_state.selected_language_type_v8, ALL_PPL_OPTIONS_CONFIG,
                                             LANGUAGE_TO_GL_MAP)  # initial_ppl_options
    st.session_state.selected_ppl_v8 = "Java" if "Java" in init_ppl_opts else (
        init_ppl_opts[0] if init_ppl_opts and init_ppl_opts[0] != "N/A" else (
            "Khác" if "Khác" in init_ppl_opts else (ALL_PPL_OPTIONS_CONFIG[0] if ALL_PPL_OPTIONS_CONFIG else "N/A")))
    st.session_state.app_v8_lang_init_done = True

# Khởi tạo session state cho "Loại Dự Án Chính"
if 'ui_selected_project_type' not in st.session_state:
    st.session_state.ui_selected_project_type = PROJECT_TYPES_OPTIONS_UI[
        0] if PROJECT_TYPES_OPTIONS_UI else 'Phát triển mới'


def on_language_type_change():
    st.session_state.selected_language_type_v8 = st.session_state.language_type_widget_key
    # (Logic cập nhật PPL dựa trên LT giữ nguyên)
    current_available_ppls = get_filtered_ppl_options(st.session_state.selected_language_type_v8,
                                                      ALL_PPL_OPTIONS_CONFIG, LANGUAGE_TO_GL_MAP)
    if st.session_state.selected_ppl_v8 not in current_available_ppls:
        if "Java" in current_available_ppls:
            st.session_state.selected_ppl_v8 = "Java"
        elif current_available_ppls and current_available_ppls[0] != "N/A":
            st.session_state.selected_ppl_v8 = current_available_ppls[0]
        else:
            st.session_state.selected_ppl_v8 = "Khác" if "Khác" in current_available_ppls else (
                ALL_PPL_OPTIONS_CONFIG[0] if ALL_PPL_OPTIONS_CONFIG else "N/A")


def on_ppl_change():
    st.session_state.selected_ppl_v8 = st.session_state.ppl_widget_key
    # (Logic cập nhật LT dựa trên PPL giữ nguyên)
    required_lt = LANGUAGE_TO_GL_MAP.get(st.session_state.selected_ppl_v8, 'Khác')
    if st.session_state.selected_ppl_v8 == 'ABAP' and '4GL' in LANGUAGE_TYPES_OPTIONS_UI: required_lt = '4GL'
    if required_lt != st.session_state.selected_language_type_v8 and required_lt in LANGUAGE_TYPES_OPTIONS_UI:
        st.session_state.selected_language_type_v8 = required_lt


with st.sidebar:
    st.header("📊 Nhập Thông tin Dự án")
    st.markdown("---")
    size_metric_choice = st.selectbox("Chỉ số kích thước đầu vào:", ('LOC', 'FP', 'UCP'), key='size_metric_choice_v8')
    def_val, st_val = (10000.0, 1000.0) if size_metric_choice == 'LOC' else (
    200.0, 10.0) if size_metric_choice == 'FP' else (100.0, 5.0)  # default_val, step_val
    size_metric_value = st.number_input(f"Nhập giá trị {size_metric_choice}:", min_value=0.0, value=def_val,
                                        step=st_val, key='size_metric_value_v8', format="%.2f")

    # ---- MỤC CHỌN "LOẠI DỰ ÁN CHÍNH" ----
    st.markdown("---")
    # Đảm bảo PROJECT_TYPES_OPTIONS_UI có giá trị trước khi dùng
    current_project_type_options = PROJECT_TYPES_OPTIONS_UI if PROJECT_TYPES_OPTIONS_UI else PROJECT_TYPES_OPTIONS_UI_DEFAULT

    # Kiểm tra nếu st.session_state.ui_selected_project_type không có trong options thì reset
    if st.session_state.ui_selected_project_type not in current_project_type_options and current_project_type_options:
        st.session_state.ui_selected_project_type = current_project_type_options[0]
    elif not current_project_type_options:  # Trường hợp cực hiếm options rỗng
        st.session_state.ui_selected_project_type = "Phát triển mới"  # Fallback an toàn
        current_project_type_options = ["Phát triển mới"]

    selected_project_type_from_ui = st.selectbox(
        "Loại Dự Án Chính:",
        options=current_project_type_options,
        index=current_project_type_options.index(
            st.session_state.ui_selected_project_type) if st.session_state.ui_selected_project_type in current_project_type_options else 0,
        key="ui_project_type_selector_widget_key",  # Key cho widget
        help="Ảnh hưởng đến tham số COCOMO cho COCOMO II và suy luận Thời gian/Nhóm cho ML."
    )
    st.session_state.ui_selected_project_type = selected_project_type_from_ui  # Cập nhật session state
    # ---- KẾT THÚC MỤC CHỌN "LOẠI DỰ ÁN CHÍNH" ----

    st.markdown("---")
    st.subheader("📋 Thông tin Chi Tiết cho Model ML (nếu có)")
    input_values_for_ml_sidebar = {}
    _current_selected_ppl_for_conversion_and_ml = st.session_state.selected_ppl_v8

    if load_successful_global and preprocessor_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        for feature_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            if feature_name == 'Project Type':  # Xử lý đặc biệt cho Project Type nếu là feature ML
                input_values_for_ml_sidebar[feature_name] = selected_project_type_from_ui  # Lấy từ mục chọn chính
                st.text(f"Loại Dự Án (cho ML): {selected_project_type_from_ui} (từ lựa chọn chung)")
                continue  # Không tạo selectbox trùng lặp

            if feature_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                if feature_name not in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:
                    input_values_for_ml_sidebar[feature_name] = st.number_input(f"{feature_name} (ML):", value=0.0,
                                                                                format="%.2f",
                                                                                key=f"ml_num_{feature_name}_v8")
                else:
                    input_values_for_ml_sidebar[feature_name] = 0.0
            elif feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                options_for_feature = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get(feature_name, [])
                # Xử lý Language Type, PPL như trước
                if feature_name == 'Language Type':
                    lt_idx = LANGUAGE_TYPES_OPTIONS_UI.index(
                        st.session_state.selected_language_type_v8) if st.session_state.selected_language_type_v8 in LANGUAGE_TYPES_OPTIONS_UI else 0
                    st.selectbox(f"{feature_name} (ML):", options=LANGUAGE_TYPES_OPTIONS_UI, index=lt_idx,
                                 key="language_type_widget_key", on_change=on_language_type_change)
                    input_values_for_ml_sidebar[feature_name] = st.session_state.selected_language_type_v8
                elif feature_name == 'Primary Programming Language':
                    # (Logic PPL selectbox giữ nguyên)
                    current_ppl_opts = get_filtered_ppl_options(st.session_state.selected_language_type_v8,
                                                                ALL_PPL_OPTIONS_CONFIG, LANGUAGE_TO_GL_MAP)
                    if st.session_state.selected_ppl_v8 not in current_ppl_opts:  # Tự sửa nếu cần
                        if "Java" in current_ppl_opts:
                            st.session_state.selected_ppl_v8 = "Java"
                        elif current_ppl_opts and current_ppl_opts[0] != "N/A":
                            st.session_state.selected_ppl_v8 = current_ppl_opts[0]
                        else:
                            st.session_state.selected_ppl_v8 = "Khác"  # Fallback
                    ppl_idx = current_ppl_opts.index(
                        st.session_state.selected_ppl_v8) if st.session_state.selected_ppl_v8 in current_ppl_opts else 0
                    st.selectbox(f"{feature_name} (ML & Quy đổi):", options=current_ppl_opts, index=ppl_idx,
                                 key="ppl_widget_key", on_change=on_ppl_change)
                    input_values_for_ml_sidebar[feature_name] = st.session_state.selected_ppl_v8
                    _current_selected_ppl_for_conversion_and_ml = st.session_state.selected_ppl_v8
                else:  # Các feature category khác
                    cat_opts = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get(feature_name, [])  # options_for_feature
                    def_cat_val = cat_opts[0] if cat_opts else None  # default_val_cat
                    cat_idx = cat_opts.index(
                        def_cat_val) if def_cat_val and cat_opts and def_cat_val in cat_opts else 0  # default_idx_cat
                    s_val = st.selectbox(f"{feature_name} (ML):", options=cat_opts if cat_opts else ["N/A"],
                                         index=cat_idx, key=f"ml_cat_{feature_name}_v8")  # sel_val
                    input_values_for_ml_sidebar[feature_name] = s_val if cat_opts else None
    else:  # Fallback nếu preprocessor không tải được - không tạo các input ML chi tiết
        st.info("Không thể tải cấu hình ML chi tiết. Sử dụng các giá trị mặc định/tổng quát.")
        # Vẫn cho phép chọn Language Type và PPL cơ bản
        lt_idx = LANGUAGE_TYPES_OPTIONS_UI.index(
            st.session_state.selected_language_type_v8) if st.session_state.selected_language_type_v8 in LANGUAGE_TYPES_OPTIONS_UI else 0
        st.selectbox("Language Type:", options=LANGUAGE_TYPES_OPTIONS_UI, index=lt_idx, key="language_type_widget_key",
                     on_change=on_language_type_change)
        # input_values_for_ml_sidebar['Language Type'] = st.session_state.selected_language_type_v8 # Không gán vào dict ML nếu không chắc là feature

        current_ppl_opts = get_filtered_ppl_options(st.session_state.selected_language_type_v8, ALL_PPL_OPTIONS_CONFIG,
                                                    LANGUAGE_TO_GL_MAP)
        ppl_idx = current_ppl_opts.index(
            st.session_state.selected_ppl_v8) if st.session_state.selected_ppl_v8 in current_ppl_opts else 0
        st.selectbox("Ngôn ngữ chính (Quy đổi):", options=current_ppl_opts, index=ppl_idx, key="ppl_widget_key",
                     on_change=on_ppl_change)
        _current_selected_ppl_for_conversion_and_ml = st.session_state.selected_ppl_v8
        # input_values_for_ml_sidebar['Primary Programming Language'] = st.session_state.selected_ppl_v8 # Tương tự

    (calc_loc, calc_fp, calc_ucp, est_effort_pm_basic, est_dev_time_basic, est_team_size_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, _current_selected_ppl_for_conversion_and_ml)

    auto_calc_inputs = {'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
                        'Development Time (months)': est_dev_time_basic,
                        'Team Size': est_team_size_basic}  # auto_calculated_numerical_inputs
    for k, v in auto_calc_inputs.items():  # key, val
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and k in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and k in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
            input_values_for_ml_sidebar[k] = v

    st.markdown("---");
    st.write("**Giá trị số cho ML (tự động từ quy đổi):**")
    # (Giữ nguyên logic hiển thị auto_filled_display_cols)
    auto_filled_display_cols = {}
    check_cols_disp = ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR if load_successful_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR else auto_calc_inputs.keys()
    for col_n in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:  # col_name
        if col_n in check_cols_disp:
            val_disp = input_values_for_ml_sidebar.get(col_n, auto_calc_inputs.get(col_n))  # value_to_display
            if val_disp is not None: auto_filled_display_cols[col_n] = f"{val_disp:,.1f}" if isinstance(val_disp, (
            float, int)) else str(val_disp)
    if auto_filled_display_cols:
        num_dc = len(auto_filled_display_cols)  # num_disp_cols
        if num_dc > 0:
            disp_cols_met = st.columns(min(num_dc, 3));
            i = 0  # disp_cols_metrics, idx
            for k_dc, v_dc in auto_filled_display_cols.items():  # k_disp, v_disp
                disp_cols_met[i % 3].metric(label=f"{k_dc} (cho ML)", value=v_dc);
                i += 1  # delta_color="off"
    else:
        st.caption("Không có giá trị số tự động cho ML để hiển thị.")

    st.markdown("---");
    st.subheader(f"📈 Kích thước Ước tính ( '{_current_selected_ppl_for_conversion_and_ml}'):")
    mc1, mc2, mc3 = st.columns(3)  # m_col1..
    mc1.metric("LOC", f"{calc_loc:,.0f}");
    mc2.metric("FP", f"{calc_fp:,.0f}");
    mc3.metric("UCP", f"{calc_ucp:,.0f}")
    st.markdown("---");
    st.subheader(f"⏱️ COCOMO Cơ bản ( '{_current_selected_ppl_for_conversion_and_ml}'):")
    mc4, mc5, mc6 = st.columns(3)  # m_col4..
    mc4.metric("Effort (PM)", f"{est_effort_pm_basic:,.1f}");
    mc5.metric("T.Gian P.T (Tháng)", f"{est_dev_time_basic:,.1f}");
    mc6.metric("Quy mô Nhóm", f"{est_team_size_basic:,.1f}")
    st.markdown("---")
    predict_disabled = not (
                load_successful_global and preprocessor_loaded_global and ml_models_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and FEATURE_NAMES_AFTER_PROCESSING)

    if st.button("🚀 Ước tính Nỗ lực Tổng hợp", key='predict_btn_v8', disabled=predict_disabled):
        final_ml_input_dict = {}  # final_input_dict_for_ml
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            for col_n in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # col_name
                final_ml_input_dict[col_n] = input_values_for_ml_sidebar.get(col_n)
                if col_n in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                    try:
                        curr_val = final_ml_input_dict[col_n]  # current_val
                        if curr_val is None: st.error(f"Lỗi: '{col_n}' là None."); st.stop()
                        final_ml_input_dict[col_n] = float(curr_val)
                    except:
                        st.error(f"Lỗi: Giá trị '{final_ml_input_dict[col_n]}' cho '{col_n}' không hợp lệ."); st.stop()
                elif col_n in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR and final_ml_input_dict[col_n] is None:
                    opts = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get(col_n, [])  # options
                    final_ml_input_dict[col_n] = "Khác" if "Khác" in opts else (opts[0] if opts else "N/A")
        else:
            st.error("Lỗi: Thiếu cấu hình cột ML."); st.stop()

        raw_ml_df = pd.DataFrame([final_ml_input_dict])  # input_df_raw_ml
        try:
            if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR: raw_ml_df = raw_ml_df[
                ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR]
            st.session_state.raw_input_df_display = raw_ml_df.copy()
        except Exception as e:
            st.error(f"Lỗi DataFrame đầu vào: {e}"); st.stop()

        proc_df_model = pd.DataFrame();
        ml_ok = False  # processed_df_for_model, ml_processing_ok
        if preprocessor_loaded_global and not raw_ml_df.empty:
            try:
                input_proc_arr = preprocessor_loaded_global.transform(raw_ml_df)  # input_processed_np_array
                if FEATURE_NAMES_AFTER_PROCESSING and len(FEATURE_NAMES_AFTER_PROCESSING) == input_proc_arr.shape[1]:
                    proc_df_model = pd.DataFrame(input_proc_arr, columns=FEATURE_NAMES_AFTER_PROCESSING)
                    st.session_state.processed_input_df_display = proc_df_model.copy();
                    ml_ok = True
                else:
                    st.error(f"Lỗi ML: Số tên đặc trưng không khớp.")
            except Exception as e:
                st.error(f"Lỗi preprocessor: {e}")
        # else: st.warning("Không thể xử lý cho mô hình ML.") # Đã có predict_disabled

        # Lấy giá trị Loại Dự Án từ widget đã chọn ở sidebar
        project_type_for_calc = st.session_state.ui_selected_project_type  # Đã được cập nhật từ selected_project_type_from_ui

        project_type_map = {'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached",
                            'Khác': "Organic"}  # project_type_for_cocomo_map
        cocomo_mode = project_type_map.get(project_type_for_calc, "Organic")  # cocomo_mode_for_time_and_effort_calc

        results_list = []
        if ml_ok and not proc_df_model.empty and ml_models_loaded_global:
            for model_n, model_o in ml_models_loaded_global.items():  # model_name, model_obj
                eff_ph_ml, dev_t_ml, team_s_ml = "Lỗi", "Lỗi", "Lỗi"  # effort_ph_ml, dev_time_ml, team_size_ml
                if model_o:
                    try:
                        pred_ph = model_o.predict(proc_df_model)[0]
                        eff_ph_ml = round(float(pred_ph), 0)
                        dev_t_ml, team_s_ml = calculate_dev_time_team_from_effort_ph(eff_ph_ml, cocomo_mode,
                                                                                     HOURS_PER_PERSON_MONTH)
                    except Exception as e:
                        eff_ph_ml = f"Lỗi dự đoán: {type(e).__name__}"
                else:
                    eff_ph_ml = "Mô hình chưa tải"
                results_list.append({'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_n,
                                     'Effort (Person-Hours)': eff_ph_ml, 'Development Time (months)': dev_t_ml,
                                     'Team Size': team_s_ml})
        # (Phần xử lý lỗi ML giữ nguyên)
        elif ml_models_loaded_global and not ml_ok and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # If ML was expected but processing failed
            for model_n_key in ml_models_loaded_global.keys(): results_list.append(
                {'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_n_key,
                 'Effort (Person-Hours)': "Lỗi dữ liệu/xử lý ML", 'Development Time (months)': "N/A",
                 'Team Size': "N/A"})
        elif not ml_models_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # If ML was expected but no models loaded
            results_list.append({'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': "Không có Model ML",
                                 'Effort (Person-Hours)': "N/A", 'Development Time (months)': "N/A",
                                 'Team Size': "N/A"})

        kloc_cocomo2 = calc_loc / 1000  # kloc_cocomo_ii
        eff_pm_coco2, dev_t_coco2, team_s_coco2 = estimate_cocomo_ii_full(kloc_cocomo2,
                                                                          project_type_cocomo=cocomo_mode)  # effort_pm_cocomo_ii, ...
        eff_ph_coco2 = round(eff_pm_coco2 * HOURS_PER_PERSON_MONTH, 0)  # effort_ph_cocomo_ii
        results_list.append({'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': "COCOMO II",
                             'Effort (Person-Hours)': eff_ph_coco2 if eff_pm_coco2 > 0 else (
                                 "N/A" if kloc_cocomo2 <= 0 else "0"),
                             'Development Time (months)': dev_t_coco2 if eff_pm_coco2 > 0 else (
                                 "N/A" if kloc_cocomo2 <= 0 else "0"),
                             'Team Size': team_s_coco2 if eff_pm_coco2 > 0 else ("N/A" if kloc_cocomo2 <= 0 else "0")})

        if not results_list:
            st.warning("Không có kết quả ước tính nào.")
        else:
            st.session_state.results_summary_df = pd.DataFrame(results_list); st.success("Đã hoàn thành ước tính!")

    if predict_disabled: st.warning("Nút 'Ước tính' bị vô hiệu hóa do thiếu tài nguyên/cấu hình ML.")

main_area_results = st.container()
with main_area_results:
    st.header("📊 Bảng Tổng Kết Ước Tính Nỗ Lực")
    if not st.session_state.results_summary_df.empty:
        # (Giữ nguyên logic hiển thị DataFrame và biểu đồ)
        st.dataframe(st.session_state.results_summary_df.style.format({
            'LOC': "{:,.0f}", 'FP': "{:,.0f}", 'UCP': "{:,.0f}",
            'Effort (Person-Hours)': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'Development Time (months)': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x),
            'Team Size': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x),
        }), use_container_width=True)
        st.subheader("📈 Biểu đồ So sánh Effort (Person-Hours)")
        df_chart = st.session_state.results_summary_df.copy()  # df_for_chart
        df_chart['Effort (Person-Hours)'] = pd.to_numeric(df_chart['Effort (Person-Hours)'], errors='coerce')
        df_chart.dropna(subset=['Effort (Person-Hours)'], inplace=True)
        df_chart = df_chart[df_chart['Effort (Person-Hours)'] > 0].sort_values(by='Effort (Person-Hours)',
                                                                               ascending=False)
        if not df_chart.empty:
            fig, ax = plt.subplots(figsize=(10, max(6, len(df_chart) * 0.6)))  # fig_compare, ax_compare
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # bar_colors
            bars = ax.bar(df_chart['Model Name'], df_chart['Effort (Person-Hours)'],
                          color=[colors[i % len(colors)] for i in range(len(df_chart))])  # bars_compare
            max_eff = df_chart['Effort (Person-Hours)'].max() if not df_chart[
                'Effort (Person-Hours)'].empty else 1  # max_effort_val
            for bar in bars:  # bar_item
                yval = bar.get_height()  # y_val_bar
                ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * max_eff, f'{yval:,.0f}', ha='center',
                        va='bottom', fontsize=9)
            ax.set_ylabel('Effort Ước tính (Person-Hours)');
            ax.set_xlabel('Mô hình Ước tính');
            ax.set_title('So sánh Effort giữa các Mô hình')
            plt.xticks(rotation=45, ha="right");
            plt.yticks();
            ax.grid(axis='y', linestyle='--', alpha=0.7);
            plt.tight_layout();
            st.pyplot(fig)
        else:
            st.info("Không có dữ liệu Effort hợp lệ để vẽ biểu đồ.")
    else:
        # (Thông báo lỗi/info giữ nguyên)
        if not load_successful_global and any(os.path.exists(p) for p in
                                              [PREPROCESSOR_PATH, FEATURES_PATH] + [mp for mp in MODEL_PATHS.values() if
                                                                                    mp]):
            st.error("Tải tài nguyên ban đầu thất bại. Ước tính ML có thể không thực hiện được.")
        else:
            st.info("Nhập thông tin và nhấn 'Ước tính Nỗ lực Tổng hợp' để xem kết quả.")

# (Phần Debug có thể giữ lại nếu bạn cần)