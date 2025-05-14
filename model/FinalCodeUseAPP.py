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
COCOMO_A = 2.4
COCOMO_B = 1.05
COCOMO_C = 2.5
COCOMO_D = 0.38

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
# Ensure all PPLs have a mapping, if 'Scripting' is a PPL:
# if 'Scripting' not in LANGUAGE_TO_GL_MAP:
#     LANGUAGE_TO_GL_MAP['Scripting'] = 'Scripting'

# --- Định nghĩa đường dẫn ---
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

# --- Khởi tạo biến cấu hình ---
ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = []
NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
FEATURE_NAMES_AFTER_PROCESSING = []
CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = {}

BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(list(AVG_LOC_PER_FP.keys()))
# Ensure 'Khác' is in PPL options if not already
if 'Khác' not in BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI:
    BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI.append('Khác')

PROJECT_TYPES_OPTIONS_UI = ['Phát triển mới', 'Nâng cấp lớn', 'Khác']
# LANGUAGE_TYPES_OPTIONS_UI will be derived from LANGUAGE_TO_GL_MAP values
LANGUAGE_TYPES_OPTIONS_UI = sorted(list(set(val for val in LANGUAGE_TO_GL_MAP.values() if val)))
if not LANGUAGE_TYPES_OPTIONS_UI: LANGUAGE_TYPES_OPTIONS_UI = ['Khác']

COUNT_APPROACH_OPTIONS_UI = ['IFPUG', 'NESMA', 'FiSMA', 'COSMIC', 'Mark II', 'LOC Manual', 'Khác']
APPLICATION_GROUP_OPTIONS_UI = ['Nghiệp vụ (Business)', 'Hỗ trợ Quyết định (Decision Support)', 'Khác']
APPLICATION_TYPES_OPTIONS_UI = ['Ứng dụng Web', 'Ứng dụng Di động', 'Ứng dụng Desktop', 'Khác']
DEVELOPMENT_TYPES_OPTIONS_UI = ['Nội bộ (In-house)', 'Thuê ngoài (Outsource)', 'Hỗn hợp (Hybrid)', 'Khác']


# --- Hàm Tính Toán ---
def calculate_metrics(size_metric_choice, size_metric_value, language):
    calculated_loc = 0.0
    calculated_fp = 0.0
    calculated_ucp = 0.0
    estimated_effort_pm = 0.0
    estimated_dev_time_months = 0.0
    estimated_team_size = 0.0
    loc_fp_ratio = AVG_LOC_PER_FP.get(language, AVG_LOC_PER_FP.get('Khác', 50))

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
            params_organic = COCOMO_II_PARAMS_BY_MODE["Organic"]
            if params_organic["a"] > 0 and params_organic["b"] != 0 and effort_pm_from_ucp > 0:
                base_cocomo_val = effort_pm_from_ucp / params_organic["a"]
                if base_cocomo_val > 0:
                    kloc_from_ucp_effort = base_cocomo_val ** (1 / params_organic["b"])
                    calculated_loc = kloc_from_ucp_effort * 1000
                    if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio
            else:
                calculated_loc = 0
                calculated_fp = 0

    if size_metric_choice != 'UCP':
        if size_metric_choice == 'LOC' and loc_fp_ratio > 0:
            calculated_fp = calculated_loc / loc_fp_ratio
        _kloc_for_ucp_calc = calculated_loc / 1000
        if _kloc_for_ucp_calc > 0:
            params_organic = COCOMO_II_PARAMS_BY_MODE["Organic"]
            _effort_pm_for_ucp_calc = params_organic["a"] * (_kloc_for_ucp_calc ** params_organic["b"])
            if EFFORT_PER_UCP > 0:
                calculated_ucp = (_effort_pm_for_ucp_calc * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
            else:
                calculated_ucp = 0
        else:
            calculated_ucp = 0

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
    if kloc <= 0: return 0.0, 0.0, 0.0
    params = COCOMO_II_PARAMS_BY_MODE.get(project_type_cocomo, COCOMO_II_PARAMS_BY_MODE["Organic"])
    a, b, c_mode, d_mode = params["a"], params["b"], params["c"], params["d"]
    effort_pm = a * (kloc ** b) * effort_multipliers_product
    dev_time_months = 0
    if effort_pm > 0: dev_time_months = c_mode * (effort_pm ** d_mode)
    team_size = (effort_pm / dev_time_months) if dev_time_months > 0 else (1 if effort_pm > 0 else 0)
    return round(effort_pm, 2), round(dev_time_months, 2), round(team_size, 1)


def calculate_dev_time_team_from_effort_ph(effort_ph, cocomo_c_const, cocomo_d_const, hrs_per_month_const):
    if effort_ph <= 0 or hrs_per_month_const <= 0: return 0.0, 0.0
    effort_pm = effort_ph / hrs_per_month_const
    dev_time_months = 0
    if effort_pm > 0: dev_time_months = cocomo_c_const * (effort_pm ** cocomo_d_const)
    team_size = (effort_pm / dev_time_months) if dev_time_months > 0 else (1 if effort_pm > 0 else 0)
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

    try:
        if not os.path.exists(PREPROCESSOR_PATH): raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}")
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
        num_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'num')
        cat_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'cat')
        extracted_numerical_features_raw = list(num_transformer_tuple[2])
        extracted_categorical_features_raw = list(cat_transformer_tuple[2])
        extracted_original_cols_order = extracted_numerical_features_raw + extracted_categorical_features_raw
        cat_pipeline = loaded_preprocessor.named_transformers_['cat']
        onehot_encoder = cat_pipeline.named_steps['onehot']
        if hasattr(onehot_encoder, 'categories_') and onehot_encoder.categories_:
            for i, feature_name in enumerate(extracted_categorical_features_raw):
                if i < len(onehot_encoder.categories_):
                    extracted_categorical_options[feature_name] = onehot_encoder.categories_[i].tolist()
                else:  # Should not happen if preprocessor is correctly built
                    extracted_categorical_options[feature_name] = []
        else:  # Fallback if categories_ is not as expected
            for feature_name in extracted_categorical_features_raw:
                extracted_categorical_options[feature_name] = []
            # st.sidebar.warning("OneHotEncoder categories not found or mismatched. Using empty lists for categorical options.")

    except Exception as e:
        # st.sidebar.error(f"Error loading preprocessor or extracting config: {e}") # Commented out for cleaner UI on initial load fail
        all_loaded_successfully = False

    try:
        if not os.path.exists(FEATURES_PATH): raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")
        loaded_feature_names_after_processing = joblib.load(FEATURES_PATH)
        if isinstance(loaded_feature_names_after_processing, np.ndarray):
            loaded_feature_names_after_processing = loaded_feature_names_after_processing.tolist()
        if not isinstance(loaded_feature_names_after_processing, list):
            loaded_feature_names_after_processing = list(loaded_feature_names_after_processing)
    except Exception as e:
        # st.sidebar.error(f"Error loading feature names: {e}")
        all_loaded_successfully = False

    models_actually_loaded = 0
    for name, path in MODEL_PATHS.items():
        try:
            if not os.path.exists(path): raise FileNotFoundError(f"Model {name} not found at {path}")
            loaded_ml_models[name] = joblib.load(path)
            models_actually_loaded += 1
        except Exception as e:
            # st.sidebar.warning(f"Error loading model {name}: {e}")
            loaded_ml_models[name] = None
    if models_actually_loaded == 0 and MODEL_PATHS: all_loaded_successfully = False  # If models were expected but none loaded

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

# Update UI options based on loaded artifacts
if load_successful_global and categorical_options_global:
    CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = categorical_options_global
    ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = original_cols_order_global
    NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = numerical_features_raw_global
    CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = categorical_features_raw_global
    FEATURE_NAMES_AFTER_PROCESSING = feature_names_loaded_global

    PROJECT_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Project Type', PROJECT_TYPES_OPTIONS_UI)
    # For Language Type, we use LANGUAGE_TO_GL_MAP as the source of truth for consistency with filtering logic.
    # If 'Language Type' is a feature in preprocessor, its options might be a subset or different.
    # We will stick to LANGUAGE_TYPES_OPTIONS_UI derived from LANGUAGE_TO_GL_MAP.
    # If preprocessor has 'Language Type' options, they are available in CATEGORICAL_OPTIONS_FROM_PREPROCESSOR['Language Type']
    # but not directly used to populate the selectbox to maintain filtering integrity.

    COUNT_APPROACH_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Count Approach', COUNT_APPROACH_OPTIONS_UI)
    APPLICATION_GROUP_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Application Group',
                                                                             APPLICATION_GROUP_OPTIONS_UI)
    APPLICATION_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Application Type',
                                                                             APPLICATION_TYPES_OPTIONS_UI)
    DEVELOPMENT_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Development Type',
                                                                             DEVELOPMENT_TYPES_OPTIONS_UI)

# --- Tiêu đề ứng dụng ---
st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v8")

# --- Session State Initialization for Cascading Selectboxes and other states ---
if 'results_summary_df' not in st.session_state: st.session_state.results_summary_df = pd.DataFrame()
if 'processed_input_df_display' not in st.session_state: st.session_state.processed_input_df_display = None
if 'raw_input_df_display' not in st.session_state: st.session_state.raw_input_df_display = None

# Determine all PPL options (from preprocessor or base list)
ALL_PPL_OPTIONS_CONFIG = BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI  # Default
if load_successful_global and 'Primary Programming Language' in CATEGORICAL_OPTIONS_FROM_PREPROCESSOR:
    ppl_from_preprocessor = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR['Primary Programming Language']
    if ppl_from_preprocessor:  # Ensure it's not empty
        ALL_PPL_OPTIONS_CONFIG = sorted(list(set(ppl_from_preprocessor)))
        if 'Khác' not in ALL_PPL_OPTIONS_CONFIG: ALL_PPL_OPTIONS_CONFIG.append('Khác')  # Ensure Khác is an option


# Helper function to get filtered PPL options based on Language Type
def get_filtered_ppl_options(language_type, all_ppls, lang_to_gl_map):
    if not language_type or language_type == "Tất cả":  # Assuming "Tất cả" could be an option for LT
        return sorted(list(set(all_ppls)))

    filtered = [lang for lang in all_ppls if lang_to_gl_map.get(lang, 'Khác') == language_type]

    if not filtered:  # If no specific PPLs match the LT
        # Fallback: show PPLs that are generally 'Khác' if LT is 'Khác'
        if language_type == 'Khác':
            khac_ppls = [lang for lang in all_ppls if lang_to_gl_map.get(lang, 'Khác') == 'Khác']
            if khac_ppls: return sorted(list(set(khac_ppls)))
        # If still no PPLs, or LT is not 'Khác' but yields no PPLs, return a default list
        return ["Khác"] if "Khác" in all_ppls else (all_ppls[:1] if all_ppls else ["N/A"])
    return sorted(list(set(filtered)))


# Initialize session state for language selection if not already done
if 'app_v8_lang_init_done' not in st.session_state:
    default_lt = '3GL' if '3GL' in LANGUAGE_TYPES_OPTIONS_UI else LANGUAGE_TYPES_OPTIONS_UI[0]
    st.session_state.selected_language_type_v8 = default_lt

    initial_ppl_options = get_filtered_ppl_options(default_lt, ALL_PPL_OPTIONS_CONFIG, LANGUAGE_TO_GL_MAP)
    default_ppl = "Java"
    if default_ppl not in initial_ppl_options:
        if initial_ppl_options and initial_ppl_options[0] != "N/A":
            default_ppl = initial_ppl_options[0]
        elif "Khác" in initial_ppl_options:
            default_ppl = "Khác"
        else:  # Ultimate fallback
            default_ppl = ALL_PPL_OPTIONS_CONFIG[0] if ALL_PPL_OPTIONS_CONFIG else "Khác"

    st.session_state.selected_ppl_v8 = default_ppl
    st.session_state.app_v8_lang_init_done = True


# --- Callbacks for language selection ---
def on_language_type_change():
    new_lt = st.session_state.language_type_widget_key  # Get value from widget that triggered
    st.session_state.selected_language_type_v8 = new_lt

    current_available_ppls = get_filtered_ppl_options(new_lt, ALL_PPL_OPTIONS_CONFIG, LANGUAGE_TO_GL_MAP)

    # If current selected PPL is not valid for the new LT, update PPL
    if st.session_state.selected_ppl_v8 not in current_available_ppls:
        if "Java" in current_available_ppls:
            st.session_state.selected_ppl_v8 = "Java"
        elif current_available_ppls and current_available_ppls[0] != "N/A":
            st.session_state.selected_ppl_v8 = current_available_ppls[0]
        elif "Khác" in current_available_ppls:
            st.session_state.selected_ppl_v8 = "Khác"
        else:  # Fallback if current_available_ppls is empty or only "N/A"
            st.session_state.selected_ppl_v8 = ALL_PPL_OPTIONS_CONFIG[0] if ALL_PPL_OPTIONS_CONFIG else "Khác"


def on_ppl_change():
    new_ppl = st.session_state.ppl_widget_key  # Get value from widget
    st.session_state.selected_ppl_v8 = new_ppl

    # Determine required LT for this PPL
    required_lt = LANGUAGE_TO_GL_MAP.get(new_ppl, 'Khác')  # Default to 'Khác' if PPL not in map
    if new_ppl == 'ABAP':
        required_lt = '4GL'
    elif new_ppl == 'SQL':
        required_lt = 'Ngôn ngữ truy vấn (SQL)'

    # If the required LT is different and valid, update selected_language_type_v8
    if required_lt != st.session_state.selected_language_type_v8 and required_lt in LANGUAGE_TYPES_OPTIONS_UI:
        st.session_state.selected_language_type_v8 = required_lt
        # This will cause the Language Type widget to update on rerun.
        # The PPL options will also be re-filtered based on this new LT.
        # The on_language_type_change logic should ensure selected_ppl_v8 remains valid if possible.


# --- Sidebar ---
with st.sidebar:
    st.header("📊 Nhập Thông tin Dự án")
    st.markdown("---")

    size_metric_choice = st.selectbox(
        "Chỉ số kích thước đầu vào:", ('LOC', 'FP', 'UCP'), key='size_metric_choice_v8'
    )
    default_val, step_val = (10000.0, 1000.0) if size_metric_choice == 'LOC' else \
        (200.0, 10.0) if size_metric_choice == 'FP' else (100.0, 5.0)
    size_metric_value = st.number_input(
        f"Nhập giá trị {size_metric_choice}:", min_value=0.0, value=default_val, step=step_val,
        key='size_metric_value_v8', format="%.2f"
    )

    st.markdown("---")
    st.subheader("📋 Thông tin Chi Tiết cho Model ML")
    input_values_for_ml_sidebar = {}

    _current_selected_ppl_for_conversion_and_ml = st.session_state.selected_ppl_v8  # Initial value from session state

    if load_successful_global and preprocessor_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        for feature_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            if feature_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                if feature_name not in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:
                    input_values_for_ml_sidebar[feature_name] = st.number_input(
                        f"{feature_name} (ML):", value=0.0, format="%.2f", key=f"ml_num_{feature_name}_v8"
                    )
                else:
                    input_values_for_ml_sidebar[feature_name] = 0.0  # Will be auto-filled

            elif feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                options_for_feature = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get(feature_name, [])

                if feature_name == 'Language Type':
                    # Determine default index for LT selectbox from session state
                    lt_default_idx = 0
                    if st.session_state.selected_language_type_v8 in LANGUAGE_TYPES_OPTIONS_UI:
                        lt_default_idx = LANGUAGE_TYPES_OPTIONS_UI.index(st.session_state.selected_language_type_v8)
                    else:  # If session state value not in options (e.g. options changed), reset to first
                        if LANGUAGE_TYPES_OPTIONS_UI:
                            st.session_state.selected_language_type_v8 = LANGUAGE_TYPES_OPTIONS_UI[0]
                        # lt_default_idx remains 0

                    st.selectbox(
                        f"{feature_name} (ML):",
                        options=LANGUAGE_TYPES_OPTIONS_UI,
                        index=lt_default_idx,
                        key="language_type_widget_key",  # Distinct key for the widget
                        on_change=on_language_type_change,
                        help="Loại ngôn ngữ. Lựa chọn này sẽ lọc danh sách Ngôn ngữ lập trình chính."
                    )
                    input_values_for_ml_sidebar[feature_name] = st.session_state.selected_language_type_v8

                elif feature_name == 'Primary Programming Language':
                    # Get current PPL options based on selected Language Type from session state
                    current_ppl_widget_options = get_filtered_ppl_options(
                        st.session_state.selected_language_type_v8,
                        ALL_PPL_OPTIONS_CONFIG,
                        LANGUAGE_TO_GL_MAP
                    )

                    # Ensure selected_ppl_v8 is valid for these options. If not, on_language_type_change should have fixed it.
                    # Or, if fixed by on_ppl_change forcing an LT, this list should now be compatible.
                    if st.session_state.selected_ppl_v8 not in current_ppl_widget_options:
                        # This can happen if callbacks lead to a state where selected PPL isn't in the filtered list for selected LT.
                        # Attempt to self-correct or use a safe default.
                        if "Java" in current_ppl_widget_options:
                            st.session_state.selected_ppl_v8 = "Java"
                        elif current_ppl_widget_options and current_ppl_widget_options[0] != "N/A":
                            st.session_state.selected_ppl_v8 = current_ppl_widget_options[0]
                        elif "Khác" in current_ppl_widget_options:
                            st.session_state.selected_ppl_v8 = "Khác"
                        else:  # Ultimate fallback
                            st.session_state.selected_ppl_v8 = ALL_PPL_OPTIONS_CONFIG[
                                0] if ALL_PPL_OPTIONS_CONFIG else "Khác"
                            if st.session_state.selected_ppl_v8 not in current_ppl_widget_options:  # If even this is not in options
                                current_ppl_widget_options = [st.session_state.selected_ppl_v8] + [opt for opt in
                                                                                                   current_ppl_widget_options
                                                                                                   if
                                                                                                   opt != st.session_state.selected_ppl_v8]

                    ppl_default_idx = 0
                    if st.session_state.selected_ppl_v8 in current_ppl_widget_options:
                        ppl_default_idx = current_ppl_widget_options.index(st.session_state.selected_ppl_v8)
                    elif current_ppl_widget_options and current_ppl_widget_options[
                        0] != "N/A":  # If not found, default to first valid
                        st.session_state.selected_ppl_v8 = current_ppl_widget_options[0]
                        # ppl_default_idx remains 0

                    st.selectbox(
                        f"{feature_name} (ML & Quy đổi):",
                        options=current_ppl_widget_options,
                        index=ppl_default_idx,
                        key="ppl_widget_key",  # Distinct key for the widget
                        on_change=on_ppl_change,
                        help="Ngôn ngữ chính. Thay đổi sẽ cập nhật Loại ngôn ngữ nếu cần. Lựa chọn bị lọc bởi Loại ngôn ngữ."
                    )
                    input_values_for_ml_sidebar[feature_name] = st.session_state.selected_ppl_v8
                    _current_selected_ppl_for_conversion_and_ml = st.session_state.selected_ppl_v8

                else:  # Other categorical features
                    default_val_cat = options_for_feature[0] if options_for_feature else None
                    # Try to get a saved value from session_state if implementing persistence for other fields
                    # key_cat = f"ml_cat_{feature_name}_v8"
                    # if key_cat in st.session_state: default_val_cat = st.session_state[key_cat]

                    default_idx_cat = 0
                    if default_val_cat and options_for_feature and default_val_cat in options_for_feature:
                        default_idx_cat = options_for_feature.index(default_val_cat)

                    sel_val = st.selectbox(
                        f"{feature_name} (ML):",
                        options=options_for_feature if options_for_feature else ["N/A"],
                        index=default_idx_cat,
                        key=f"ml_cat_{feature_name}_v8"
                    )
                    input_values_for_ml_sidebar[feature_name] = sel_val if options_for_feature else None
    else:  # Fallback if preprocessor not loaded
        st.warning("Lỗi tải tài nguyên ML. Các trường nhập liệu ML và quy đổi có thể không chính xác hoặc bị hạn chế.")
        # Fallback for PPL and LT selection
        # Language Type
        lt_fb_default_idx = 0
        if st.session_state.selected_language_type_v8 in LANGUAGE_TYPES_OPTIONS_UI:
            lt_fb_default_idx = LANGUAGE_TYPES_OPTIONS_UI.index(st.session_state.selected_language_type_v8)

        st.selectbox(
            "Language Type (ML):",
            options=LANGUAGE_TYPES_OPTIONS_UI,
            index=lt_fb_default_idx,
            key="language_type_widget_key",
            on_change=on_language_type_change,
            help="Loại ngôn ngữ. Lựa chọn này sẽ lọc danh sách Ngôn ngữ lập trình chính."
        )
        if 'Language Type' in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR or not ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # Add if expected or if no expectation set
            input_values_for_ml_sidebar['Language Type'] = st.session_state.selected_language_type_v8

        # Primary Programming Language
        current_ppl_fb_options = get_filtered_ppl_options(st.session_state.selected_language_type_v8,
                                                          ALL_PPL_OPTIONS_CONFIG, LANGUAGE_TO_GL_MAP)
        ppl_fb_default_idx = 0
        if st.session_state.selected_ppl_v8 in current_ppl_fb_options:
            ppl_fb_default_idx = current_ppl_fb_options.index(st.session_state.selected_ppl_v8)
        elif current_ppl_fb_options and current_ppl_fb_options[0] != "N/A":
            st.session_state.selected_ppl_v8 = current_ppl_fb_options[0]

        st.selectbox(
            "Ngôn ngữ chính (ML & Quy đổi):",
            options=current_ppl_fb_options,
            index=ppl_fb_default_idx,
            key="ppl_widget_key",
            on_change=on_ppl_change,
            help="Ngôn ngữ chính của dự án. Dùng cho cả mô hình ML và quy đổi LOC/FP/UCP."
        )
        if 'Primary Programming Language' in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR or not ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            input_values_for_ml_sidebar['Primary Programming Language'] = st.session_state.selected_ppl_v8
        _current_selected_ppl_for_conversion_and_ml = st.session_state.selected_ppl_v8

    # Recalculate metrics based on the potentially updated PPL from session state
    (calc_loc, calc_fp, calc_ucp,
     est_effort_pm_basic, est_dev_time_basic, est_team_size_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, _current_selected_ppl_for_conversion_and_ml
    )

    auto_calculated_numerical_inputs = {
        'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
        'Development Time (months)': est_dev_time_basic,
        'Team Size': est_team_size_basic
    }
    # Update input_values_for_ml_sidebar with these auto-calculated values if they are expected features
    # This ensures that even if these fields are not manually entered widgets, their values are part of the ML input
    for key, val in auto_calculated_numerical_inputs.items():
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # Only if we know the expected columns
            if key in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and key in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                input_values_for_ml_sidebar[key] = val
        # If preprocessor not loaded, these might not be in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR
        # but we still might want them for COCOMO II or other parts.
        # The current structure correctly puts them into input_values_for_ml_sidebar if they are expected num features.

    # Display auto-filled numerical inputs for ML
    if load_successful_global or not ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # Show if loaded or if no specific expectations
        st.markdown("---")
        st.write("**Giá trị số cho ML (tự động tính từ quy đổi):**")
        auto_filled_display_cols = {}
        # Check against ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR if available
        check_cols = ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR if load_successful_global else auto_calculated_numerical_inputs.keys()

        for col_name in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:
            if col_name in check_cols:  # Only display if it's an expected feature or universally relevant
                # Use value from input_values_for_ml_sidebar if it was set there, otherwise from auto_calculated_numerical_inputs
                value_to_display = input_values_for_ml_sidebar.get(col_name,
                                                                   auto_calculated_numerical_inputs.get(col_name))
                if value_to_display is not None:
                    auto_filled_display_cols[col_name] = f"{value_to_display:,.1f}" if isinstance(value_to_display, (
                    float, int)) else str(value_to_display)

        if auto_filled_display_cols:
            num_disp_cols = len(auto_filled_display_cols)
            if num_disp_cols > 0:
                disp_cols_metrics = st.columns(min(num_disp_cols, 3))  # Max 3 columns for display
                idx = 0
                for k, v in auto_filled_display_cols.items():
                    disp_cols_metrics[idx % 3].metric(label=f"{k} (cho ML)", value=v, delta_color="off")
                    idx += 1
        else:
            st.caption("Không có giá trị số tự động nào được cấu hình để hiển thị cho ML.")

    st.markdown("---")
    st.subheader(f"📈 Kích thước Ước tính (cho '{_current_selected_ppl_for_conversion_and_ml}'):")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("LOC", f"{calc_loc:,.0f}", delta_color="off")
    m_col2.metric("FP", f"{calc_fp:,.0f}", delta_color="off")
    m_col3.metric("UCP", f"{calc_ucp:,.0f}", delta_color="off")

    st.markdown("---")
    st.subheader(f"⏱️ COCOMO Cơ bản (cho '{_current_selected_ppl_for_conversion_and_ml}'):")
    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("Effort (PM)", f"{est_effort_pm_basic:,.1f}", delta_color="off")
    m_col5.metric("T.Gian P.T (Tháng)", f"{est_dev_time_basic:,.1f}", delta_color="off")
    m_col6.metric("Quy mô Nhóm", f"{est_team_size_basic:,.1f}", delta_color="off")

    st.markdown("---")
    # Predict button should be enabled if essential components for prediction are ready
    # For ML models, preprocessor and models need to be loaded.
    # For COCOMO II, it can run independently of ML models if calc_loc is available.
    predict_disabled = not (
                load_successful_global and preprocessor_loaded_global and ml_models_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR)

    if st.button("🚀 Ước tính Nỗ lực Tổng hợp", key='predict_btn_v8', disabled=predict_disabled):
        final_input_dict_for_ml = {}
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            for col_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
                final_input_dict_for_ml[col_name] = input_values_for_ml_sidebar.get(col_name)
                # Ensure numerical features are float
                if col_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                    try:
                        current_val = final_input_dict_for_ml[col_name]
                        if current_val is None:  # Handle None if a numerical field wasn't populated
                            st.error(f"Lỗi: Giá trị cho '{col_name}' là None. Cần một giá trị số.")
                            st.stop()
                        final_input_dict_for_ml[col_name] = float(current_val)
                    except (ValueError, TypeError):
                        st.error(
                            f"Lỗi: Giá trị cho '{col_name}' ('{final_input_dict_for_ml[col_name]}') không phải là số hợp lệ.")
                        st.stop()
                # Ensure categorical features are not None (use "Khác" or a placeholder if necessary)
                elif col_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                    if final_input_dict_for_ml[col_name] is None:
                        # Try to get options for this feature to see if "Khác" is valid
                        options = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get(col_name, [])
                        if "Khác" in options:
                            final_input_dict_for_ml[col_name] = "Khác"
                        elif options:
                            final_input_dict_for_ml[col_name] = options[0]  # First available option
                        else:
                            final_input_dict_for_ml[col_name] = "N/A"  # Absolute fallback
                        # st.warning(f"Giá trị cho '{col_name}' là None, đã thay thế bằng '{final_input_dict_for_ml[col_name]}'.")

        else:  # Should not happen if predict_disabled logic is correct
            st.error("Lỗi: Không có thông tin cột mong đợi từ preprocessor. Không thể chuẩn bị dữ liệu cho ML.")
            st.stop()

        input_df_raw_ml = pd.DataFrame([final_input_dict_for_ml])
        try:
            if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
                input_df_raw_ml = input_df_raw_ml[ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR]
            st.session_state.raw_input_df_display = input_df_raw_ml.copy()
        except KeyError as e:
            st.error(
                f"Lỗi sắp xếp cột cho preprocessor: {e}. Kiểm tra ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR và dữ liệu đầu vào: {final_input_dict_for_ml.keys()}")
            st.stop()
        except Exception as e_general:
            st.error(f"Lỗi DataFrame đầu vào thô: {e_general}")
            st.stop()

        processed_df_for_model = pd.DataFrame()
        ml_processing_ok = False
        if preprocessor_loaded_global and not input_df_raw_ml.empty:
            try:
                input_processed_np_array = preprocessor_loaded_global.transform(input_df_raw_ml)
                if FEATURE_NAMES_AFTER_PROCESSING and len(FEATURE_NAMES_AFTER_PROCESSING) == \
                        input_processed_np_array.shape[1]:
                    processed_df_for_model = pd.DataFrame(input_processed_np_array,
                                                          columns=FEATURE_NAMES_AFTER_PROCESSING)
                    st.session_state.processed_input_df_display = processed_df_for_model.copy()
                    ml_processing_ok = True
                else:
                    st.error(
                        f"Lỗi ML: Số tên đặc trưng sau xử lý ({len(FEATURE_NAMES_AFTER_PROCESSING)}) không khớp với số cột đầu ra ({input_processed_np_array.shape[1]}).")
            except Exception as e_proc:
                st.error(f"Lỗi áp dụng preprocessor: {e_proc}")
                # st.error(traceback.format_exc()) # Detailed traceback for debugging
        else:
            st.warning("Preprocessor/dữ liệu ML trống hoặc không thể tải. Không thể xử lý cho mô hình ML.")

        results_list = []
        if ml_processing_ok and not processed_df_for_model.empty and ml_models_loaded_global:
            for model_name, model_obj in ml_models_loaded_global.items():
                effort_ph_ml, dev_time_ml, team_size_ml = "Lỗi", "Lỗi", "Lỗi"
                if model_obj:
                    try:
                        pred_ph = model_obj.predict(processed_df_for_model)
                        effort_ph_ml = round(float(pred_ph[0]), 0)
                        dev_time_ml, team_size_ml = calculate_dev_time_team_from_effort_ph(
                            effort_ph_ml, COCOMO_C, COCOMO_D, HOURS_PER_PERSON_MONTH
                        )
                    except Exception as e_pred:
                        effort_ph_ml = f"Lỗi dự đoán ({type(e_pred).__name__})"
                else:
                    effort_ph_ml = "Mô hình chưa tải"
                results_list.append({
                    'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_name,
                    'Effort (Person-Hours)': effort_ph_ml, 'Development Time (months)': dev_time_ml,
                    'Team Size': team_size_ml
                })
        else:  # ML processing failed or no models
            if ml_models_loaded_global and not ml_processing_ok:  # Models exist but processing failed
                for model_name_key in ml_models_loaded_global.keys():
                    results_list.append({
                        'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_name_key,
                        'Effort (Person-Hours)': "Lỗi dữ liệu/xử lý ML", 'Development Time (months)': "N/A",
                        'Team Size': "N/A"
                    })
            elif not ml_models_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # No models loaded but ML path was expected
                results_list.append({
                    'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': "Không có Model ML nào được tải",
                    'Effort (Person-Hours)': "N/A", 'Development Time (months)': "N/A", 'Team Size': "N/A"
                })

        # COCOMO II Calculation (always attempted if calc_loc is available)
        kloc_cocomo_ii = calc_loc / 1000
        project_type_for_cocomo_map = {'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached", 'Khác': "Organic"}

        # Get project type for COCOMO II. Use 'Project Type' from ML inputs if available, otherwise default.
        project_type_val_for_cocomo = 'Phát triển mới'  # Default
        if 'Project Type' in input_values_for_ml_sidebar:
            project_type_val_for_cocomo = input_values_for_ml_sidebar['Project Type']
        elif 'Project Type' in PROJECT_TYPES_OPTIONS_UI:  # Fallback to first option if not in sidebar dict
            project_type_val_for_cocomo = PROJECT_TYPES_OPTIONS_UI[0]

        cocomo_mode_calc = project_type_for_cocomo_map.get(project_type_val_for_cocomo, "Organic")

        effort_pm_cocomo_ii, dev_time_cocomo_ii, team_size_cocomo_ii = estimate_cocomo_ii_full(kloc_cocomo_ii,
                                                                                               project_type_cocomo=cocomo_mode_calc)
        effort_ph_cocomo_ii = round(effort_pm_cocomo_ii * HOURS_PER_PERSON_MONTH, 0)

        results_list.append({
            'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': "COCOMO II",
            'Effort (Person-Hours)': effort_ph_cocomo_ii if effort_pm_cocomo_ii > 0 else (
                "N/A" if kloc_cocomo_ii <= 0 else "0"),  # Show 0 if kloc > 0 but effort is 0
            'Development Time (months)': dev_time_cocomo_ii if effort_pm_cocomo_ii > 0 else (
                "N/A" if kloc_cocomo_ii <= 0 else "0"),
            'Team Size': team_size_cocomo_ii if effort_pm_cocomo_ii > 0 else ("N/A" if kloc_cocomo_ii <= 0 else "0")
        })

        if not results_list:  # If somehow no results were generated (e.g. no ML models and COCOMO failed)
            st.warning("Không có kết quả ước tính nào được tạo. Vui lòng kiểm tra đầu vào và cấu hình.")
        else:
            st.session_state.results_summary_df = pd.DataFrame(results_list)
            st.success("Đã hoàn thành ước tính tổng hợp!")

    if predict_disabled:
        st.warning(
            "Nút 'Ước tính' bị vô hiệu hóa do thiếu tài nguyên ML cần thiết (preprocessor, models, feature names, hoặc cấu hình cột). Vui lòng kiểm tra bảng điều khiển để biết thêm chi tiết nếu có lỗi tải.")

# --- Khu vực chính hiển thị kết quả ---
main_area_results = st.container()
with main_area_results:
    st.header("📊 Bảng Tổng Kết Ước Tính Nỗ Lực")
    if not st.session_state.results_summary_df.empty:
        st.dataframe(st.session_state.results_summary_df.style.format({
            'LOC': "{:,.0f}", 'FP': "{:,.0f}", 'UCP': "{:,.0f}",
            'Effort (Person-Hours)': lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x),
            'Development Time (months)': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x),
            'Team Size': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x),
        }), use_container_width=True)

        st.subheader("📈 Biểu đồ So sánh Effort (Person-Hours)")
        df_for_chart = st.session_state.results_summary_df.copy()
        # Convert to numeric, coercing errors. Rows with non-numeric effort will be dropped for chart.
        df_for_chart['Effort (Person-Hours)'] = pd.to_numeric(df_for_chart['Effort (Person-Hours)'], errors='coerce')
        df_for_chart.dropna(subset=['Effort (Person-Hours)'], inplace=True)
        df_for_chart = df_for_chart[df_for_chart['Effort (Person-Hours)'] > 0]  # Only plot positive efforts
        df_for_chart = df_for_chart.sort_values(by='Effort (Person-Hours)', ascending=False)

        if not df_for_chart.empty:
            fig_compare, ax_compare = plt.subplots(
                figsize=(10, max(6, len(df_for_chart) * 0.6)))  # Adjusted height factor
            bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                          '#bcbd22', '#17becf']
            bars_compare = ax_compare.bar(df_for_chart['Model Name'], df_for_chart['Effort (Person-Hours)'],
                                          color=[bar_colors[i % len(bar_colors)] for i in range(len(df_for_chart))])

            max_effort_val = df_for_chart['Effort (Person-Hours)'].max() if not df_for_chart[
                'Effort (Person-Hours)'].empty else 1
            for bar_item in bars_compare:
                y_val_bar = bar_item.get_height()
                # Adjust text position to be slightly above the bar
                ax_compare.text(bar_item.get_x() + bar_item.get_width() / 2.0, y_val_bar + 0.01 * max_effort_val,
                                f'{y_val_bar:,.0f}', ha='center', va='bottom', fontsize=9, color='black')

            ax_compare.set_ylabel('Effort Ước tính (Person-Hours)', fontsize=12)
            ax_compare.set_xlabel('Mô hình Ước tính', fontsize=12)  # Added X-axis label
            ax_compare.set_title('So sánh Effort giữa các Mô hình', fontsize=14)  # Added Chart Title
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            ax_compare.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            st.pyplot(fig_compare)
        else:
            st.info("Không có đủ dữ liệu Effort hợp lệ (>0) để vẽ biểu đồ so sánh.")
    else:
        if not load_successful_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # Only show error if ML was expected
            st.error(
                "Tải tài nguyên ban đầu thất bại. Không thể thực hiện ước tính ML. COCOMO II có thể vẫn hoạt động nếu có kích thước LOC.")
        else:
            st.info("Nhập thông tin ở thanh bên và nhấn '🚀 Ước tính Nỗ lực Tổng hợp' để xem kết quả.")

# --- Hiển thị thông tin debug (tùy chọn, có thể xóa hoặc ẩn đi) ---
# with st.expander("Thông tin Debug (Trạng thái Session và Cấu hình)"):
#     st.write("Trạng thái Session (Ngôn ngữ):")
#     st.json({
#         "selected_language_type_v8": st.session_state.get("selected_language_type_v8"),
#         "selected_ppl_v8": st.session_state.get("selected_ppl_v8"),
#         "language_type_widget_key (raw)": st.session_state.get("language_type_widget_key"),
#         "ppl_widget_key (raw)": st.session_state.get("ppl_widget_key"),
#         "app_v8_lang_init_done": st.session_state.get("app_v8_lang_init_done")
#     })
#     st.write("Cấu hình PPL và LT:")
#     st.json({
#         "ALL_PPL_OPTIONS_CONFIG": ALL_PPL_OPTIONS_CONFIG,
#         "LANGUAGE_TYPES_OPTIONS_UI": LANGUAGE_TYPES_OPTIONS_UI,
#         "LANGUAGE_TO_GL_MAP": LANGUAGE_TO_GL_MAP
#     })
#     st.write("Thông tin Preprocessor (nếu đã tải):")
#     st.json({
#         "load_successful_global": load_successful_global,
#         "ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR": ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR,
#         "CATEGORICAL_OPTIONS_FROM_PREPROCESSOR": CATEGORICAL_OPTIONS_FROM_PREPROCESSOR if load_successful_global else "Chưa tải"
#     })
#     if st.session_state.get('raw_input_df_display') is not None:
#         st.write("Dữ liệu thô cho ML:")
#         st.dataframe(st.session_state.raw_input_df_display)
#     if st.session_state.get('processed_input_df_display') is not None:
#         st.write("Dữ liệu đã xử lý cho ML:")
#         st.dataframe(st.session_state.processed_input_df_display)
