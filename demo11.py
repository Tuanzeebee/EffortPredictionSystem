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
COCOMO_A = 2.4  # Thường dùng cho Organic mode, a
COCOMO_B = 1.05  # Thường dùng cho Organic mode, b
COCOMO_C = 2.5  # Thường dùng cho Organic mode, c (cho Development Time)
COCOMO_D = 0.38  # Thường dùng cho Organic mode, d (cho Development Time)

# Tham số COCOMO II cho các mode khác nhau (bao gồm cả c, d cho Development Time)
# Bạn có thể cần điều chỉnh các giá trị c, d này cho chính xác hơn theo tài liệu COCOMO II
COCOMO_II_PARAMS_BY_MODE = {
    "Organic": {"a": 2.4, "b": 1.05, "c": 2.5, "d": 0.38},
    "Semi-detached": {"a": 3.0, "b": 1.12, "c": 2.5, "d": 0.35},  # c, d là ví dụ, cần kiểm tra
    "Embedded": {"a": 3.6, "b": 1.20, "c": 2.5, "d": 0.32}  # c, d là ví dụ, cần kiểm tra
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

# Mapping ngôn ngữ sang loại ngôn ngữ (3GL, 4GL, etc.)
# Đảm bảo các giá trị (ví dụ: '3GL') khớp với những gì OneHotEncoder đã học cho cột 'Language Type'
LANGUAGE_TO_GL_MAP = {
    'Java': '3GL', 'Python': '3GL', 'C++': '3GL', 'C#': '3GL',
    'JavaScript': 'Scripting',  # Hoặc '3GL' tùy theo định nghĩa của bạn
    'SQL': 'Ngôn ngữ truy vấn (SQL)',
    'COBOL': '3GL',  # COBOL thường được coi là 3GL
    'ABAP': '4GL',
    'PHP': 'Scripting',  # Hoặc '3GL'
    'Swift': '3GL', 'Kotlin': '3GL', 'Ruby': 'Scripting', 'Go': '3GL',
    'Assembly': 'Assembly',
    'Scripting': 'Scripting',  # Cho các ngôn ngữ scripting chung
    'Visual Basic': '3GL',  # Hoặc '4GL' tùy phiên bản/cách nhìn
    'Ada': '3GL',
    'Perl': 'Scripting',
    'Khác': 'Khác'  # Giá trị mặc định
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
LANGUAGE_TYPES_OPTIONS_UI = ['3GL', '4GL', 'Scripting', 'Ngôn ngữ truy vấn (SQL)', 'Assembly',
                             'Khác']  # Cập nhật để chứa giá trị từ map
PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(list(AVG_LOC_PER_FP.keys()))
# ... (các OPTIONS_UI khác giữ nguyên hoặc cập nhật nếu cần) ...
COUNT_APPROACH_OPTIONS_UI = ['IFPUG', 'NESMA', 'FiSMA', 'COSMIC', 'Mark II', 'LOC Manual', 'Khác']
APPLICATION_GROUP_OPTIONS_UI = ['Nghiệp vụ (Business)', 'Hỗ trợ Quyết định (Decision Support)',
                                'Khoa học/Kỹ thuật (Scientific/Engineering)', 'Thời gian thực (Real-time)',
                                'Hệ thống (System Software)', 'Tiện ích (Utility)', 'Khác']
APPLICATION_TYPES_OPTIONS_UI = ['Ứng dụng Web', 'Ứng dụng Di động', 'Ứng dụng Desktop', 'Hệ thống Nhúng',
                                'Xử lý Dữ liệu/Batch', 'API/Dịch vụ', 'Trí tuệ nhân tạo/ML', 'Game', 'Khác']
DEVELOPMENT_TYPES_OPTIONS_UI = ['Nội bộ (In-house)', 'Thuê ngoài (Outsource)', 'Hỗn hợp (Hybrid)',
                                'Mã nguồn mở (Đóng góp)', 'Sản phẩm (COTS) tùy chỉnh', 'Khác']


# --- Hàm Tính Toán ---
def calculate_metrics(size_metric_choice, size_metric_value, language):
    # ... (Giữ nguyên logic hàm này) ...
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
            # Sử dụng COCOMO_A, COCOMO_B (mặc định là Organic) để quy đổi ngược ra KLOC
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
    # Sử dụng tham số COCOMO (c,d) mặc định (Organic) cho ước tính COCOMO Basic ban đầu
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


# --- Hàm COCOMO II (Cập nhật để trả về Effort PM, Dev Time, Team Size) ---
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
    elif effort_pm > 0:  # Nếu effort > 0 nhưng dev_time = 0 (ví dụ do effort quá nhỏ)
        team_size = 1

    return round(effort_pm, 2), round(dev_time_months, 2), round(team_size, 1)


# --- Hàm tính Dev Time và Team Size từ Effort (Person-Hours) cho ML ---
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
        else:  # If effort_pm is very small, dev_time_months might become 0 or very close
            team_size = 1 if effort_pm > 0 else 0

    return round(dev_time_months, 2), round(team_size, 1)


# --- Hàm tải mô hình, preprocessor và trích xuất cấu hình ---
@st.cache_resource
def load_artifacts_and_extract_config():
    # ... (Giữ nguyên logic hàm này, chỉ cần đảm bảo các st.sidebar.write/caption không bị xóa nếu bạn muốn giữ chúng) ...
    loaded_preprocessor = None
    loaded_feature_names_after_processing = []
    loaded_ml_models = OrderedDict()
    extracted_original_cols_order = []
    extracted_numerical_features_raw = []
    extracted_categorical_features_raw = []
    extracted_categorical_options = {}
    all_loaded_successfully = True

    if not os.path.exists(PREPROCESSOR_PATH):
        # st.sidebar.error(f"LỖI: Không tìm thấy preprocessor tại '{PREPROCESSOR_PATH}'") # Bị loại bỏ theo yêu cầu
        all_loaded_successfully = False
    else:
        try:
            loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
            # st.sidebar.write("✔️ Preprocessor đã tải.") # Bị loại bỏ
            try:
                num_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'num')
                cat_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'cat')
                extracted_numerical_features_raw = list(num_transformer_tuple[2])
                extracted_categorical_features_raw = list(cat_transformer_tuple[2])
                extracted_original_cols_order = extracted_numerical_features_raw + extracted_categorical_features_raw
                # st.sidebar.caption(f"Preprocessor: {len(extracted_numerical_features_raw)} cột số, {len(extracted_categorical_features_raw)} cột loại.") # Bị loại bỏ

                cat_pipeline = loaded_preprocessor.named_transformers_['cat']
                onehot_encoder = cat_pipeline.named_steps['onehot']
                if hasattr(onehot_encoder, 'categories_') and len(onehot_encoder.categories_) == len(
                        extracted_categorical_features_raw):
                    for i, feature_name in enumerate(extracted_categorical_features_raw):
                        categories = onehot_encoder.categories_[i].tolist()
                        extracted_categorical_options[feature_name] = categories
                    # st.sidebar.write("✔️ Tùy chọn trường phân loại đã trích xuất.") # Bị loại bỏ
                else:
                    # st.sidebar.error("Lỗi trích xuất: 'categories_' từ OneHotEncoder không hợp lệ.") # Bị loại bỏ
                    all_loaded_successfully = False
            except Exception:  # as e_extract:
                # st.sidebar.error(f"Lỗi trích xuất cấu hình từ preprocessor: {e_extract}") # Bị loại bỏ
                all_loaded_successfully = False
        except Exception:  # as e_load_prep:
            # st.sidebar.error(f"Lỗi tải preprocessor: {e_load_prep}") # Bị loại bỏ
            all_loaded_successfully = False

    if not os.path.exists(FEATURES_PATH):
        # st.sidebar.error(f"LỖI: Không tìm thấy feature_names tại '{FEATURES_PATH}'") # Bị loại bỏ
        all_loaded_successfully = False
    else:
        try:
            loaded_feature_names_after_processing = joblib.load(FEATURES_PATH)
            if isinstance(loaded_feature_names_after_processing, np.ndarray):
                loaded_feature_names_after_processing = loaded_feature_names_after_processing.tolist()
            if not isinstance(loaded_feature_names_after_processing, list):
                loaded_feature_names_after_processing = list(loaded_feature_names_after_processing)
            # st.sidebar.write(f"✔️ Tên đặc trưng sau xử lý ({len(loaded_feature_names_after_processing)} cột) đã tải.") # Bị loại bỏ
        except Exception:  # as e_load_feat:
            # st.sidebar.error(f"Lỗi tải feature names: {e_load_feat}") # Bị loại bỏ
            all_loaded_successfully = False

    models_actually_loaded = 0
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            # st.sidebar.warning(f"Cảnh báo: Không tìm thấy mô hình '{name}' tại '{path}'.") # Bị loại bỏ
            loaded_ml_models[name] = None
            continue
        try:
            model = joblib.load(path)
            loaded_ml_models[name] = model
            models_actually_loaded += 1
        except Exception:  # as e_load_model:
            # st.sidebar.warning(f"Lỗi tải mô hình {name}: {e_load_model}.") # Bị loại bỏ
            loaded_ml_models[name] = None
    if models_actually_loaded == 0:
        # st.sidebar.error("LỖI: Không tải được mô hình ML nào.") # Bị loại bỏ
        all_loaded_successfully = False
    # else:
    # st.sidebar.write(f"✔️ Đã tải {models_actually_loaded}/{len(MODEL_PATHS)} mô hình ML.") # Bị loại bỏ

    # if not all_loaded_successfully:
    # st.sidebar.error("Tải tài nguyên ML thất bại.") # Bị loại bỏ

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
 categorical_options_global,
 load_successful_global
 ) = load_artifacts_and_extract_config()

# Cập nhật các biến UI options nếu tải thành công
if load_successful_global and categorical_options_global:
    PROJECT_TYPES_OPTIONS_UI = categorical_options_global.get('Project Type', PROJECT_TYPES_OPTIONS_UI)
    # LANGUAGE_TYPES_OPTIONS_UI sẽ được cập nhật để bao gồm các giá trị từ map
    all_gl_types_from_map = list(set(LANGUAGE_TO_GL_MAP.values()))
    current_lang_type_opts = categorical_options_global.get('Language Type', [])
    # Kết hợp và loại bỏ trùng lặp, đảm bảo các giá trị từ map có mặt
    combined_lang_types = sorted(list(set(current_lang_type_opts + all_gl_types_from_map + LANGUAGE_TYPES_OPTIONS_UI)))
    LANGUAGE_TYPES_OPTIONS_UI = combined_lang_types

    COUNT_APPROACH_OPTIONS_UI = categorical_options_global.get('Count Approach', COUNT_APPROACH_OPTIONS_UI)
    APPLICATION_GROUP_OPTIONS_UI = categorical_options_global.get('Application Group', APPLICATION_GROUP_OPTIONS_UI)
    APPLICATION_TYPES_OPTIONS_UI = categorical_options_global.get('Application Type', APPLICATION_TYPES_OPTIONS_UI)
    DEVELOPMENT_TYPES_OPTIONS_UI = categorical_options_global.get('Development Type', DEVELOPMENT_TYPES_OPTIONS_UI)

    ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = original_cols_order_global
    NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = numerical_features_raw_global
    CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = categorical_features_raw_global
    FEATURE_NAMES_AFTER_PROCESSING = feature_names_loaded_global

# --- Tiêu đề ứng dụng ---
st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v6")

# Khởi tạo session state
if 'results_summary_df' not in st.session_state: st.session_state.results_summary_df = None
# Các session state cũ không còn cần thiết nếu không hiển thị riêng lẻ
# if 'ml_predictions_ph' not in st.session_state: st.session_state.ml_predictions_ph = None
# if 'cocomo_estimate_ph' not in st.session_state: st.session_state.cocomo_estimate_ph = None
if 'processed_input_df_display' not in st.session_state: st.session_state.processed_input_df_display = None
if 'raw_input_df_display' not in st.session_state: st.session_state.raw_input_df_display = None

# --- Sidebar ---
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

    # Ngôn ngữ cho quy đổi LOC/FP/UCP (ví dụ Java, ABAP)
    lang_idx = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI.index(
        'Java') if 'Java' in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI else 0
    selected_primary_lang_for_conversion = st.selectbox(
        "Ngôn ngữ (cho quy đổi LOC/FP/UCP):", PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI, index=lang_idx,
        key='lang_conversion_v6',
        help="Chọn ngôn ngữ chính của dự án để hỗ trợ quy đổi giữa LOC, FP, UCP."
    )

    (calc_loc, calc_fp, calc_ucp, est_effort_pm_basic, est_dev_time_basic, est_team_size_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, selected_primary_lang_for_conversion
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
    input_values_for_ml_sidebar = {}

    if load_successful_global and preprocessor_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        auto_filled_values = {
            'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
            'Development Time (months)': est_dev_time_basic,  # Sẽ được ghi đè bởi tính toán từ ML/COCOMO II trong bảng
            'Team Size': est_team_size_basic  # Sẽ được ghi đè
        }

        for feature_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            if feature_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                if feature_name in auto_filled_values:
                    input_values_for_ml_sidebar[feature_name] = auto_filled_values[feature_name]
                    # st.text(f"{feature_name} (auto): {auto_filled_values[feature_name]:,.1f}") # Giảm bớt hiển thị
                else:
                    input_values_for_ml_sidebar[feature_name] = st.number_input(
                        f"{feature_name} (ML):", value=0.0, format="%.2f", key=f"ml_num_{feature_name}"
                    )
            elif feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                options_for_feature = categorical_options_global.get(feature_name, ["Lỗi: Ko có options"])

                current_selection_cat = None
                idx_cat = 0

                if feature_name == 'Language Type':
                    # Tự động set dựa trên selected_primary_lang_for_conversion
                    suggested_lang_type = LANGUAGE_TO_GL_MAP.get(selected_primary_lang_for_conversion, 'Khác')
                    if suggested_lang_type in options_for_feature:
                        current_selection_cat = suggested_lang_type
                    elif options_for_feature and options_for_feature[0] != "Lỗi: Ko có options":
                        current_selection_cat = options_for_feature[0]  # Fallback
                    help_text_lang_type = f"Gợi ý: {suggested_lang_type} (dựa trên {selected_primary_lang_for_conversion})"
                elif feature_name == 'Primary Programming Language' and selected_primary_lang_for_conversion in options_for_feature:
                    current_selection_cat = selected_primary_lang_for_conversion
                    help_text_lang_type = None  # Không cần help text thêm
                elif options_for_feature and options_for_feature[0] != "Lỗi: Ko có options":
                    current_selection_cat = options_for_feature[0]
                    help_text_lang_type = None

                if current_selection_cat and current_selection_cat in options_for_feature:
                    try:
                        idx_cat = options_for_feature.index(current_selection_cat)
                    except ValueError:
                        idx_cat = 0

                val_selected_cat = st.selectbox(
                    f"{feature_name} (ML):", options_for_feature, index=idx_cat,
                    key=f"ml_cat_{feature_name}",
                    help=help_text_lang_type if feature_name == 'Language Type' else None
                )
                input_values_for_ml_sidebar[feature_name] = val_selected_cat.strip() if isinstance(val_selected_cat,
                                                                                                   str) else val_selected_cat
    else:
        st.warning("Lỗi tải tài nguyên ML. Không thể tạo các trường nhập liệu chi tiết.")

    st.markdown("---")
    predict_disabled = not load_successful_global
    if st.button("🚀 Ước tính Nỗ lực Tổng hợp", key='predict_btn_v6', disabled=predict_disabled):
        final_input_dict_for_ml = {}
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            for col_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
                final_input_dict_for_ml[col_name] = input_values_for_ml_sidebar.get(col_name)

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

        # DEBUG section removed as per request

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
                        f"Lỗi ML: Số tên đặc trưng ({len(FEATURE_NAMES_AFTER_PROCESSING)}) không khớp ({input_processed_np_array.shape[1]}).")
            except Exception as e_proc:
                st.error(f"Lỗi áp dụng preprocessor: {e_proc}")
                st.error(traceback.format_exc())
        else:
            st.warning("Preprocessor/dữ liệu ML trống.")

        # --- Tạo bảng tổng kết ---
        results_list = []

        # 1. Dự đoán từ các mô hình ML
        if ml_processing_ok and not processed_df_for_model.empty and ml_models_loaded_global:
            for model_name, model_obj in ml_models_loaded_global.items():
                effort_ph_ml = "Lỗi"
                dev_time_ml = "Lỗi"
                team_size_ml = "Lỗi"
                if model_obj:
                    try:
                        pred_ph = model_obj.predict(processed_df_for_model)
                        effort_ph_ml = round(float(pred_ph[0]), 0)
                        # Tính Dev Time và Team Size cho mô hình ML này
                        dev_time_ml, team_size_ml = calculate_dev_time_team_from_effort_ph(
                            effort_ph_ml, COCOMO_C, COCOMO_D, HOURS_PER_PERSON_MONTH
                        )
                    except Exception:  # as e_pred_ml:
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
        else:  # Trường hợp không xử lý được ML
            if ml_models_loaded_global:
                for model_name_key in ml_models_loaded_global.keys():
                    results_list.append({
                        'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
                        'Model Name': model_name_key,
                        'Effort (Person-Hours)': "Lỗi dữ liệu/xử lý",
                        'Development Time (months)': "N/A",
                        'Team Size': "N/A"
                    })

        # 2. Tính toán từ COCOMO II
        kloc_cocomo_ii = calc_loc / 1000
        project_type_for_cocomo_map = {'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached",
                                       'Bảo trì': "Organic", 'Tái cấu trúc': "Semi-detached",
                                       'Tích hợp hệ thống': "Embedded", 'Khác': "Organic"}
        project_type_val_for_cocomo = input_values_for_ml_sidebar.get('Project Type', 'Phát triển mới')
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

# --- Khu vực chính hiển thị kết quả ---
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

        # Biểu đồ so sánh Effort (Person-Hours)
        st.subheader("📈 Biểu đồ So sánh Effort (Person-Hours)")
        df_for_chart = st.session_state.results_summary_df.copy()
        # Chuyển đổi Effort sang số, lỗi thành NaN để vẽ biểu đồ
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
            # ax_compare.set_title('So sánh Effort Ước tính', fontsize=14) # Tiêu đề đã có ở st.subheader
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

    # Các phần hiển thị đã yêu cầu loại bỏ:
    # st.session_state.raw_input_df_display
    # st.session_state.processed_input_df_display
    # Các st.metric riêng lẻ
    # Phần Hướng dẫn và Lưu ý Quan Trọng

# Để chạy ứng dụng này:
# 1. Cài đặt các thư viện cần thiết.
# 2. Chuẩn bị các file .joblib và đặt chúng vào cùng thư mục với file script này (hoặc cập nhật OUTPUT_DIR).
# 3. Chạy lệnh: streamlit run your_script_name.py