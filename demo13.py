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
CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = {}  # Sẽ được load

# Danh sách các ngôn ngữ cơ bản cho AVG_LOC_PER_FP (dùng làm fallback nếu preprocessor không có PPL)
BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(list(AVG_LOC_PER_FP.keys()))

PROJECT_TYPES_OPTIONS_UI = ['Phát triển mới', 'Nâng cấp lớn', 'Khác']
LANGUAGE_TYPES_OPTIONS_UI = sorted(list(set(LANGUAGE_TO_GL_MAP.values())))
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
    loc_fp_ratio = AVG_LOC_PER_FP.get(language,
                                      AVG_LOC_PER_FP.get('Khác', 50))  # Thêm fallback cho AVG_LOC_PER_FP['Khác']

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

    if size_metric_choice != 'UCP':  # Tính UCP từ LOC/FP
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


# --- Hàm COCOMO II ---
def estimate_cocomo_ii_full(kloc, project_type_cocomo="Organic", effort_multipliers_product=1.0):
    if kloc <= 0: return 0.0, 0.0, 0.0
    params = COCOMO_II_PARAMS_BY_MODE.get(project_type_cocomo, COCOMO_II_PARAMS_BY_MODE["Organic"])
    a, b, c_mode, d_mode = params["a"], params["b"], params["c"], params["d"]
    effort_pm = a * (kloc ** b) * effort_multipliers_product
    dev_time_months = 0
    if effort_pm > 0: dev_time_months = c_mode * (effort_pm ** d_mode)
    team_size = (effort_pm / dev_time_months) if dev_time_months > 0 else (1 if effort_pm > 0 else 0)
    return round(effort_pm, 2), round(dev_time_months, 2), round(team_size, 1)


# --- Hàm tính Dev Time và Team Size từ Effort (Person-Hours) cho ML ---
def calculate_dev_time_team_from_effort_ph(effort_ph, cocomo_c_const, cocomo_d_const, hrs_per_month_const):
    if effort_ph <= 0 or hrs_per_month_const <= 0: return 0.0, 0.0
    effort_pm = effort_ph / hrs_per_month_const
    dev_time_months = 0
    if effort_pm > 0: dev_time_months = cocomo_c_const * (effort_pm ** cocomo_d_const)
    team_size = (effort_pm / dev_time_months) if dev_time_months > 0 else (1 if effort_pm > 0 else 0)
    return round(dev_time_months, 2), round(team_size, 1)


# --- Hàm tải mô hình, preprocessor và trích xuất cấu hình ---
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
        if hasattr(onehot_encoder, 'categories_') and len(onehot_encoder.categories_) == len(
                extracted_categorical_features_raw):
            for i, feature_name in enumerate(extracted_categorical_features_raw):
                extracted_categorical_options[feature_name] = onehot_encoder.categories_[i].tolist()
        else:
            raise ValueError("Invalid OneHotEncoder categories")
    except Exception as e:
        # st.sidebar.error(f"Error loading preprocessor or extracting config: {e}")
        all_loaded_successfully = False

    try:
        if not os.path.exists(FEATURES_PATH): raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")
        loaded_feature_names_after_processing = joblib.load(FEATURES_PATH)
        if isinstance(loaded_feature_names_after_processing, np.ndarray):
            loaded_feature_names_after_processing = loaded_feature_names_after_processing.tolist()
        if not isinstance(loaded_feature_names_after_processing, list):  # Ensure it's a list
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
    if models_actually_loaded == 0: all_loaded_successfully = False

    return (
        loaded_preprocessor, loaded_feature_names_after_processing, loaded_ml_models,
        extracted_original_cols_order, extracted_numerical_features_raw,
        extracted_categorical_features_raw, extracted_categorical_options,
        all_loaded_successfully
    )


# --- Tải tài nguyên và cấu hình một lần ---
(preprocessor_loaded_global,
 feature_names_loaded_global,
 ml_models_loaded_global,
 original_cols_order_global,
 numerical_features_raw_global,
 categorical_features_raw_global,
 categorical_options_global,  # Đổi tên để rõ ràng hơn
 load_successful_global
 ) = load_artifacts_and_extract_config()

# Cập nhật các biến UI options nếu tải thành công và gán vào CATEGORICAL_OPTIONS_FROM_PREPROCESSOR
if load_successful_global and categorical_options_global:
    CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = categorical_options_global
    ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = original_cols_order_global
    NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = numerical_features_raw_global
    CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = categorical_features_raw_global
    FEATURE_NAMES_AFTER_PROCESSING = feature_names_loaded_global

    PROJECT_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Project Type', PROJECT_TYPES_OPTIONS_UI)
    # LANGUAGE_TYPES_OPTIONS_UI đã được định nghĩa từ map, preprocessor có thể có tập con hoặc khác
    # Chúng ta sẽ dùng các options từ preprocessor cho 'Language Type' nếu nó là một feature

    # PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI sẽ lấy từ preprocessor nếu 'Primary Programming Language' là feature
    # Nếu không, dùng BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI

    COUNT_APPROACH_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Count Approach', COUNT_APPROACH_OPTIONS_UI)
    APPLICATION_GROUP_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Application Group',
                                                                             APPLICATION_GROUP_OPTIONS_UI)
    APPLICATION_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Application Type',
                                                                             APPLICATION_TYPES_OPTIONS_UI)
    DEVELOPMENT_TYPES_OPTIONS_UI = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get('Development Type',
                                                                             DEVELOPMENT_TYPES_OPTIONS_UI)

# --- Tiêu đề ứng dụng ---
st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v8")

# Khởi tạo session state
if 'results_summary_df' not in st.session_state: st.session_state.results_summary_df = pd.DataFrame()
if 'processed_input_df_display' not in st.session_state: st.session_state.processed_input_df_display = None
if 'raw_input_df_display' not in st.session_state: st.session_state.raw_input_df_display = None

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
    # Mục chọn Ngôn ngữ riêng cho quy đổi đã được loại bỏ theo yêu cầu.

    st.markdown("---")
    st.subheader("📋 Thông tin Chi Tiết cho Model ML")
    input_values_for_ml_sidebar = {}

    # Biến tạm để giữ giá trị ngôn ngữ được chọn từ PPL(ML) cho việc tính toán ngay trong sidebar
    _current_selected_ppl_for_conversion_and_ml = "Java"  # Giá trị mặc định ban đầu

    if load_successful_global and preprocessor_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        # Lấy các giá trị được tính toán ban đầu (sẽ được cập nhật sau khi PPL(ML) thay đổi)
        # Chúng ta cần PPL(ML) được chọn trước khi tính các giá trị này một cách chính xác.
        # Vòng lặp này sẽ tạo các widget, bao gồm cả PPL(ML).

        temp_ppl_value_from_widget = None  # Sẽ lưu giá trị từ PPL(ML) selectbox

        for feature_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            if feature_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                # Các trường như LOC, FP, UCP, Dev Time, Team Size nếu là input cho ML
                # sẽ được điền tự động dựa trên tính toán từ `calculate_metrics` (sau khi PPL(ML) được chọn)
                # Hiện tại, chúng ta chỉ tạo widget nếu nó không phải là các trường auto-calc này.
                if feature_name not in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:
                    input_values_for_ml_sidebar[feature_name] = st.number_input(
                        f"{feature_name} (ML):", value=0.0, format="%.2f", key=f"ml_num_{feature_name}_v8"
                    )
                else:
                    # Các trường này sẽ được điền sau, tạm thời gán None hoặc 0
                    input_values_for_ml_sidebar[feature_name] = 0.0

            elif feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                options_for_feature = CATEGORICAL_OPTIONS_FROM_PREPROCESSOR.get(feature_name, [])

                if feature_name == 'Primary Programming Language':
                    # Đây là nguồn chính cho ngôn ngữ, options từ preprocessor
                    actual_ppl_options = options_for_feature if options_for_feature else BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI
                    if not actual_ppl_options: actual_ppl_options = ["Khác"]  # Fallback cuối cùng

                    default_ppl_idx = 0
                    if _current_selected_ppl_for_conversion_and_ml in actual_ppl_options:
                        default_ppl_idx = actual_ppl_options.index(_current_selected_ppl_for_conversion_and_ml)
                    elif 'Java' in actual_ppl_options:  # Ưu tiên Java nếu có
                        default_ppl_idx = actual_ppl_options.index('Java')
                    elif actual_ppl_options:  # Nếu không thì lấy phần tử đầu tiên
                        default_ppl_idx = 0

                    # Tạo selectbox và lấy giá trị hiện tại của nó
                    selected_val = st.selectbox(
                        f"{feature_name} (ML & Quy đổi):",
                        options=actual_ppl_options,
                        index=default_ppl_idx,
                        key=f"ml_cat_{feature_name}_v8",
                        help="Ngôn ngữ chính của dự án. Dùng cho cả mô hình ML và quy đổi LOC/FP/UCP."
                    )
                    input_values_for_ml_sidebar[feature_name] = selected_val
                    temp_ppl_value_from_widget = selected_val  # Lưu lại để dùng ngay

                elif feature_name == 'Language Type':
                    # Được suy ra từ PPL(ML) đã chọn (temp_ppl_value_from_widget)
                    # Hoặc từ _current_selected_ppl_for_conversion_and_ml nếu PPL widget chưa render
                    lang_to_map_lt = temp_ppl_value_from_widget if temp_ppl_value_from_widget else _current_selected_ppl_for_conversion_and_ml
                    suggested_lt = LANGUAGE_TO_GL_MAP.get(lang_to_map_lt, 'Khác')

                    actual_lt_options = options_for_feature if options_for_feature else LANGUAGE_TYPES_OPTIONS_UI
                    if not actual_lt_options: actual_lt_options = ["Khác"]

                    final_lt = suggested_lt
                    if suggested_lt not in actual_lt_options:
                        if 'Khác' in actual_lt_options:
                            final_lt = 'Khác'
                        elif actual_lt_options:
                            final_lt = actual_lt_options[0]

                    input_values_for_ml_sidebar[feature_name] = final_lt
                    st.markdown(f"**{feature_name} (ML - từ '{lang_to_map_lt}'):** `{final_lt}`")

                else:  # Các trường categorical khác
                    default_idx = 0
                    # (Thêm logic chọn default index cho các trường khác nếu cần)
                    sel_val = st.selectbox(
                        f"{feature_name} (ML):",
                        options=options_for_feature if options_for_feature else ["N/A"],
                        index=default_idx,
                        key=f"ml_cat_{feature_name}_v8"
                    )
                    input_values_for_ml_sidebar[feature_name] = sel_val if options_for_feature else None

        # Cập nhật ngôn ngữ chính được chọn để tính toán lại các metric
        if temp_ppl_value_from_widget:
            _current_selected_ppl_for_conversion_and_ml = temp_ppl_value_from_widget

    else:  # Không load được preprocessor
        st.warning("Lỗi tải tài nguyên ML. Các trường nhập liệu ML và quy đổi có thể không chính xác.")
        # Fallback cho PPL nếu không có gì từ preprocessor
        _current_selected_ppl_for_conversion_and_ml = st.selectbox(
            "Ngôn ngữ chính (ML & Quy đổi):",
            options=BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI,
            index=BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI.index(
                "Java") if "Java" in BASE_PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI else 0,
            key="ml_cat_Primary Programming Language_v8_fallback",
            help="Ngôn ngữ chính của dự án. Dùng cho cả mô hình ML và quy đổi LOC/FP/UCP."
        )
        # Và gán giá trị này vào input_values_for_ml_sidebar nếu PPL là một feature mong đợi
        if 'Primary Programming Language' in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            input_values_for_ml_sidebar['Primary Programming Language'] = _current_selected_ppl_for_conversion_and_ml
        if 'Language Type' in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            input_values_for_ml_sidebar['Language Type'] = LANGUAGE_TO_GL_MAP.get(
                _current_selected_ppl_for_conversion_and_ml, 'Khác')

    # Tính toán các metrics dựa trên ngôn ngữ đã chọn từ PPL(ML)
    (calc_loc, calc_fp, calc_ucp,
     est_effort_pm_basic, est_dev_time_basic, est_team_size_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, _current_selected_ppl_for_conversion_and_ml
    )

    # Cập nhật các giá trị LOC, FP, UCP, Dev Time, Team Size vào input_values_for_ml_sidebar nếu chúng là feature của ML
    # Điều này xảy ra sau khi chúng được tính toán lại dựa trên PPL(ML)
    auto_calculated_numerical_inputs = {
        'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
        'Development Time (months)': est_dev_time_basic,
        'Team Size': est_team_size_basic
    }
    if load_successful_global:  # Chỉ cập nhật nếu preprocessor đã load và biết các features này
        for key, val in auto_calculated_numerical_inputs.items():
            if key in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and key in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                input_values_for_ml_sidebar[key] = val

        # Hiển thị các giá trị số được tự động điền/cập nhật cho ML
        st.markdown("---")
        st.write("**Giá trị số cho ML (tự động tính từ quy đổi):**")
        auto_filled_display_cols = {}
        for col_name in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:
            if col_name in input_values_for_ml_sidebar and \
                    col_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR and \
                    col_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                value = input_values_for_ml_sidebar[col_name]
                auto_filled_display_cols[col_name] = f"{value:,.1f}" if isinstance(value, (float, int)) else str(value)

        if auto_filled_display_cols:
            num_disp_cols = len(auto_filled_display_cols)
            if num_disp_cols > 0:
                disp_cols_metrics = st.columns(num_disp_cols)
                for i, (k, v) in enumerate(auto_filled_display_cols.items()):
                    disp_cols_metrics[i].metric(label=f"{k} (cho ML)", value=v, delta_color="off")

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
    predict_disabled = not load_successful_global
    if st.button("🚀 Ước tính Nỗ lực Tổng hợp", key='predict_btn_v8', disabled=predict_disabled):
        final_input_dict_for_ml = {}
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            for col_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
                final_input_dict_for_ml[col_name] = input_values_for_ml_sidebar.get(col_name)
                if col_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                    try:
                        final_input_dict_for_ml[col_name] = float(final_input_dict_for_ml[col_name])
                    except (ValueError, TypeError):
                        st.error(
                            f"Lỗi: Giá trị cho '{col_name}' ('{final_input_dict_for_ml[col_name]}') không phải là số hợp lệ.")
                        st.stop()
        else:
            st.error("Lỗi: Không có thông tin cột mong đợi từ preprocessor.")
            st.stop()

        input_df_raw_ml = pd.DataFrame([final_input_dict_for_ml])
        try:
            if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # Đảm bảo thứ tự cột
                input_df_raw_ml = input_df_raw_ml[ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR]
            st.session_state.raw_input_df_display = input_df_raw_ml.copy()
        except KeyError as e:
            st.error(
                f"Lỗi sắp xếp cột cho preprocessor: {e}. Kiểm tra ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR và dữ liệu đầu vào.")
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
                st.error(traceback.format_exc())
        else:
            st.warning("Preprocessor/dữ liệu ML trống hoặc không thể tải.")

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
                    except Exception:
                        effort_ph_ml = "Lỗi dự đoán"
                else:
                    effort_ph_ml = "Mô hình chưa tải"
                results_list.append({
                    'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_name,
                    'Effort (Person-Hours)': effort_ph_ml, 'Development Time (months)': dev_time_ml,
                    'Team Size': team_size_ml
                })
        else:
            if ml_models_loaded_global:  # Nếu có model nhưng xử lý lỗi
                for model_name_key in ml_models_loaded_global.keys():
                    results_list.append({
                        'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': model_name_key,
                        'Effort (Person-Hours)': "Lỗi dữ liệu/xử lý ML", 'Development Time (months)': "N/A",
                        'Team Size': "N/A"
                    })
            else:  # Không có model nào được tải
                results_list.append({
                    'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': "Không có Model ML",
                    'Effort (Person-Hours)': "N/A", 'Development Time (months)': "N/A", 'Team Size': "N/A"
                })

        kloc_cocomo_ii = calc_loc / 1000
        project_type_for_cocomo_map = {'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached",
                                       'Khác': "Organic"}  # Map rút gọn
        project_type_val_for_cocomo = input_values_for_ml_sidebar.get('Project Type', 'Phát triển mới')
        cocomo_mode_calc = project_type_for_cocomo_map.get(project_type_val_for_cocomo, "Organic")

        effort_pm_cocomo_ii, dev_time_cocomo_ii, team_size_cocomo_ii = estimate_cocomo_ii_full(kloc_cocomo_ii,
                                                                                               project_type_cocomo=cocomo_mode_calc)
        effort_ph_cocomo_ii = round(effort_pm_cocomo_ii * HOURS_PER_PERSON_MONTH, 0)
        results_list.append({
            'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp, 'Model Name': "COCOMO II",
            'Effort (Person-Hours)': effort_ph_cocomo_ii if effort_pm_cocomo_ii > 0 else "N/A",
            'Development Time (months)': dev_time_cocomo_ii if effort_pm_cocomo_ii > 0 else "N/A",
            'Team Size': team_size_cocomo_ii if effort_pm_cocomo_ii > 0 else "N/A"
        })
        st.session_state.results_summary_df = pd.DataFrame(results_list)
        st.success("Đã hoàn thành ước tính tổng hợp!")

# --- Khu vực chính hiển thị kết quả ---
main_area_results = st.container()
with main_area_results:
    st.header("📊 Bảng Tổng Kết Ước Tính Nỗ Lực")
    if not st.session_state.results_summary_df.empty:
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
            bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            bars_compare = ax_compare.bar(df_for_chart['Model Name'], df_for_chart['Effort (Person-Hours)'],
                                          color=[bar_colors[i % len(bar_colors)] for i in range(len(df_for_chart))])
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
    else:  # results_summary_df is empty
        if not load_successful_global:
            st.error("Tải tài nguyên ban đầu thất bại. Không thể thực hiện ước tính.")
        else:
            st.info("Nhập thông tin ở thanh bên và nhấn '🚀 Ước tính Nỗ lực Tổng hợp' để xem kết quả.")