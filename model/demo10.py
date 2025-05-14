# Import các thư viện cần thiết
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Để tải mô hình và preprocessors
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import traceback  # Để in lỗi chi tiết

# --- Cấu hình trang Streamlit (PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN) ---
st.set_page_config(layout="wide", page_title="Ước tính Effort PM")

# --- Hằng số và Dữ liệu Mô phỏng ---
COCOMO_A = 2.4
COCOMO_B = 1.05
COCOMO_C = 2.5
COCOMO_D = 0.38
EFFORT_PER_UCP = 20  # Person-hours per UCP
HOURS_PER_PERSON_MONTH = 152  # Số giờ làm việc trung bình mỗi tháng cho một người

AVG_LOC_PER_FP = {
    'Java': 53, 'Python': 35, 'C++': 47, 'C#': 54, 'JavaScript': 47,
    'SQL': 15, 'COBOL': 90, 'ABAP': 70, 'PHP': 40, 'Swift': 30,
    'Kotlin': 32, 'Ruby': 25, 'Go': 45, 'Assembly': 200,
    'Scripting': 20, 'Visual Basic': 32, 'Ada': 71, 'Perl': 27,
    'Khác': 50
}

# --- Định nghĩa đường dẫn cho Mô hình và Preprocessor ---
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

# --- Khởi tạo các biến cấu hình sẽ được điền từ preprocessor ---
ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = []
NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = []
FEATURE_NAMES_AFTER_PROCESSING = []
CATEGORICAL_OPTIONS_FROM_PREPROCESSOR = {}

# Các lựa chọn mặc định cho Selectbox
PROJECT_TYPES_OPTIONS_UI = ['Phát triển mới', 'Nâng cấp lớn', 'Bảo trì', 'Tái cấu trúc', 'Tích hợp hệ thống', 'Khác']
LANGUAGE_TYPES_OPTIONS_UI = ['3GL', '4GL', 'Assembly', 'Scripting', 'Ngôn ngữ truy vấn (SQL)',
                             'Ngôn ngữ đánh dấu (HTML/XML)', 'Khác']
PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI = sorted(list(AVG_LOC_PER_FP.keys()))
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
        calculated_loc = calculated_fp * loc_fp_ratio
    elif size_metric_choice == 'UCP':
        calculated_ucp = size_metric_value
        if EFFORT_PER_UCP > 0 and HOURS_PER_PERSON_MONTH > 0:
            effort_ph_from_ucp = calculated_ucp * EFFORT_PER_UCP
            effort_pm_from_ucp = effort_ph_from_ucp / HOURS_PER_PERSON_MONTH
            if COCOMO_A > 0 and COCOMO_B != 0 and effort_pm_from_ucp > 0:
                base_cocomo_val = effort_pm_from_ucp / COCOMO_A
                if base_cocomo_val > 0:
                    kloc_from_ucp_effort = base_cocomo_val ** (1 / COCOMO_B)
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
            _effort_pm_for_ucp_calc = COCOMO_A * (_kloc_for_ucp_calc ** COCOMO_B)
            if EFFORT_PER_UCP > 0:
                calculated_ucp = (_effort_pm_for_ucp_calc * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
            else:
                calculated_ucp = 0
        else:
            calculated_ucp = 0
    elif size_metric_choice == 'UCP' and calculated_loc == 0:
        pass

    final_kloc = calculated_loc / 1000
    if final_kloc > 0:
        estimated_effort_pm = COCOMO_A * (final_kloc ** COCOMO_B)
        if estimated_effort_pm > 0:
            dev_time_base_for_formula = estimated_effort_pm
            if dev_time_base_for_formula > 0:
                estimated_dev_time_months = COCOMO_C * (dev_time_base_for_formula ** COCOMO_D)
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
def estimate_cocomo_effort(kloc, project_type_cocomo="Organic", cost_drivers=None):
    effort_multipliers = 1.0
    if cost_drivers:
        for driver_value in cost_drivers.values():
            effort_multipliers *= driver_value
    cocomo_params = {"Organic": (2.4, 1.05), "Semi-detached": (3.0, 1.12), "Embedded": (3.6, 1.20)}
    a, b = cocomo_params.get(project_type_cocomo, cocomo_params["Organic"])
    if kloc <= 0: return 0.0
    effort_pm = a * (kloc ** b) * effort_multipliers
    return round(effort_pm, 2)


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

    if not os.path.exists(PREPROCESSOR_PATH):
        st.sidebar.error(f"LỖI: Không tìm thấy preprocessor tại '{PREPROCESSOR_PATH}'")
        all_loaded_successfully = False
    else:
        try:
            loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
            st.sidebar.write("✔️ Preprocessor đã tải.")
            try:
                num_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'num')
                cat_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'cat')
                extracted_numerical_features_raw = list(num_transformer_tuple[2])
                extracted_categorical_features_raw = list(cat_transformer_tuple[2])
                extracted_original_cols_order = extracted_numerical_features_raw + extracted_categorical_features_raw
                st.sidebar.caption(
                    f"Preprocessor: {len(extracted_numerical_features_raw)} cột số, {len(extracted_categorical_features_raw)} cột loại.")

                cat_pipeline = loaded_preprocessor.named_transformers_['cat']
                onehot_encoder = cat_pipeline.named_steps['onehot']
                if hasattr(onehot_encoder, 'categories_') and len(onehot_encoder.categories_) == len(
                        extracted_categorical_features_raw):
                    for i, feature_name in enumerate(extracted_categorical_features_raw):
                        categories = onehot_encoder.categories_[i].tolist()
                        extracted_categorical_options[feature_name] = categories
                    st.sidebar.write("✔️ Tùy chọn trường phân loại đã trích xuất.")
                else:
                    st.sidebar.error("Lỗi trích xuất: 'categories_' từ OneHotEncoder không hợp lệ.")
                    all_loaded_successfully = False
            except Exception as e_extract:
                st.sidebar.error(f"Lỗi trích xuất cấu hình từ preprocessor: {e_extract}")
                all_loaded_successfully = False
        except Exception as e_load_prep:
            st.sidebar.error(f"Lỗi tải preprocessor: {e_load_prep}")
            all_loaded_successfully = False

    if not os.path.exists(FEATURES_PATH):
        st.sidebar.error(f"LỖI: Không tìm thấy feature_names tại '{FEATURES_PATH}'")
        all_loaded_successfully = False
    else:
        try:
            loaded_feature_names_after_processing = joblib.load(FEATURES_PATH)
            if isinstance(loaded_feature_names_after_processing, np.ndarray):
                loaded_feature_names_after_processing = loaded_feature_names_after_processing.tolist()
            if not isinstance(loaded_feature_names_after_processing, list):
                loaded_feature_names_after_processing = list(loaded_feature_names_after_processing)
            st.sidebar.write(f"✔️ Tên đặc trưng sau xử lý ({len(loaded_feature_names_after_processing)} cột) đã tải.")
        except Exception as e_load_feat:
            st.sidebar.error(f"Lỗi tải feature names: {e_load_feat}")
            all_loaded_successfully = False

    models_actually_loaded = 0
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            st.sidebar.warning(f"Cảnh báo: Không tìm thấy mô hình '{name}' tại '{path}'.")
            loaded_ml_models[name] = None
            continue
        try:
            model = joblib.load(path)
            loaded_ml_models[name] = model
            models_actually_loaded += 1
        except Exception as e_load_model:
            st.sidebar.warning(f"Lỗi tải mô hình {name}: {e_load_model}.")
            loaded_ml_models[name] = None
    if models_actually_loaded > 0:
        st.sidebar.write(f"✔️ Đã tải {models_actually_loaded}/{len(MODEL_PATHS)} mô hình ML.")
    else:
        st.sidebar.error("LỖI: Không tải được mô hình ML nào.")
        all_loaded_successfully = False

    if not all_loaded_successfully:
        st.sidebar.error("Tải tài nguyên ML thất bại.")

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
    LANGUAGE_TYPES_OPTIONS_UI = categorical_options_global.get('Language Type', LANGUAGE_TYPES_OPTIONS_UI)
    COUNT_APPROACH_OPTIONS_UI = categorical_options_global.get('Count Approach', COUNT_APPROACH_OPTIONS_UI)
    APPLICATION_GROUP_OPTIONS_UI = categorical_options_global.get('Application Group', APPLICATION_GROUP_OPTIONS_UI)
    APPLICATION_TYPES_OPTIONS_UI = categorical_options_global.get('Application Type', APPLICATION_TYPES_OPTIONS_UI)
    DEVELOPMENT_TYPES_OPTIONS_UI = categorical_options_global.get('Development Type', DEVELOPMENT_TYPES_OPTIONS_UI)

    ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR = original_cols_order_global
    NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = numerical_features_raw_global
    CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR = categorical_features_raw_global
    FEATURE_NAMES_AFTER_PROCESSING = feature_names_loaded_global

# --- Tiêu đề ứng dụng ---
st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v5.1")

# Khởi tạo session state
if 'ml_predictions_ph' not in st.session_state: st.session_state.ml_predictions_ph = None
if 'cocomo_estimate_ph' not in st.session_state: st.session_state.cocomo_estimate_ph = None
if 'processed_input_df_display' not in st.session_state: st.session_state.processed_input_df_display = None
if 'raw_input_df_display' not in st.session_state: st.session_state.raw_input_df_display = None

# --- Sidebar ---
with st.sidebar:
    st.header("📊 Nhập Thông tin & Ước tính")
    st.markdown("---")

    size_metric_choice = st.selectbox(
        "Chỉ số kích thước đầu vào:", ('LOC', 'FP', 'UCP'), key='size_metric_choice_v5_1'
    )
    default_val, step_val = (10000.0, 1000.0) if size_metric_choice == 'LOC' else (
    200.0, 10.0) if size_metric_choice == 'FP' else (100.0, 5.0)
    size_metric_value = st.number_input(
        f"Nhập giá trị {size_metric_choice}:", min_value=0.0, value=default_val, step=step_val,
        key='size_metric_value_v5_1', format="%.2f"
    )
    lang_idx = PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI.index(
        'Java') if 'Java' in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI else 0
    selected_lang_for_conversion = st.selectbox(
        "Ngôn ngữ (cho quy đổi LOC/FP/UCP):", PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_UI, index=lang_idx,
        key='lang_conversion_v5_1'
    )

    (calc_loc, calc_fp, calc_ucp, est_effort_pm_basic, est_dev_time_basic, est_team_size_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, selected_lang_for_conversion
    )

    st.markdown("---")
    st.subheader("📈 Kích thước Ước tính:")
    col_loc, col_fp, col_ucp_sb = st.columns(3)
    col_loc.metric("LOC", f"{calc_loc:,.0f}", "Đầu vào" if size_metric_choice == 'LOC' else "Tính toán",
                   delta_color="off")
    col_fp.metric("FP", f"{calc_fp:,.0f}", "Đầu vào" if size_metric_choice == 'FP' else "Tính toán", delta_color="off")
    col_ucp_sb.metric("UCP", f"{calc_ucp:,.0f}", "Đầu vào" if size_metric_choice == 'UCP' else "Tính toán",
                      delta_color="off")

    st.markdown("---")
    st.subheader("⏱️ COCOMO Cơ bản:")
    col_e_pm_sb, col_t_m_sb, col_s_p_sb = st.columns(3)
    col_e_pm_sb.metric("Effort (PM)", f"{est_effort_pm_basic:,.1f}")
    col_t_m_sb.metric("T.Gian P.Triển (Tháng)", f"{est_dev_time_basic:,.1f}")
    col_s_p_sb.metric("Quy mô Nhóm", f"{est_team_size_basic:,.1f}")

    st.markdown("---")
    st.subheader("📋 Thông tin cho Model ML:")
    input_values_for_ml_sidebar = {}

    if load_successful_global and preprocessor_loaded_global and ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
        auto_filled_values = {
            'LOC': calc_loc, 'FP': calc_fp, 'UCP': calc_ucp,
            'Development Time (months)': est_dev_time_basic,
            'Team Size': est_team_size_basic
        }

        for feature_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            if feature_name in NUMERICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                if feature_name in auto_filled_values:
                    input_values_for_ml_sidebar[feature_name] = auto_filled_values[feature_name]
                    st.text(f"{feature_name} (auto): {auto_filled_values[feature_name]:,.1f}")
                else:
                    input_values_for_ml_sidebar[feature_name] = st.number_input(
                        f"{feature_name} (ML):", value=0.0, format="%.2f", key=f"ml_num_{feature_name}"
                    )
            elif feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR:
                options_for_feature = categorical_options_global.get(feature_name, ["Lỗi: Ko có options"])

                current_selection_cat = None
                if feature_name == 'Primary Programming Language' and selected_lang_for_conversion in options_for_feature:
                    current_selection_cat = selected_lang_for_conversion
                elif options_for_feature and options_for_feature[0] != "Lỗi: Ko có options":
                    current_selection_cat = options_for_feature[0]

                idx_cat = 0
                if current_selection_cat and current_selection_cat in options_for_feature:
                    try:
                        idx_cat = options_for_feature.index(current_selection_cat)
                    except ValueError:
                        idx_cat = 0

                val_selected_cat = st.selectbox(
                    f"{feature_name} (ML):", options_for_feature, index=idx_cat, key=f"ml_cat_{feature_name}"
                )
                input_values_for_ml_sidebar[feature_name] = val_selected_cat.strip() if isinstance(val_selected_cat,
                                                                                                   str) else val_selected_cat
    else:
        st.warning("Lỗi tải tài nguyên ML. Không thể tạo các trường nhập liệu chi tiết.")

    st.markdown("---")
    predict_disabled = not load_successful_global
    if st.button("🚀 Dự đoán Effort (ML & COCOMO II)", key='predict_btn_v5_1', disabled=predict_disabled):
        final_input_dict_for_ml = {}
        if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
            for col_name in ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:
                final_input_dict_for_ml[col_name] = input_values_for_ml_sidebar.get(col_name)

        input_df_raw_ml = pd.DataFrame([final_input_dict_for_ml])
        try:
            if ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR:  # Kiểm tra list không rỗng
                input_df_raw_ml = input_df_raw_ml[ORIGINAL_COLS_ORDER_EXPECTED_BY_PREPROCESSOR]  # Sửa lỗi chính tả
            st.session_state.raw_input_df_display = input_df_raw_ml.copy()
        except KeyError as e:
            st.error(
                f"Lỗi sắp xếp cột cho preprocessor: {e}. Kiểm tra xem tất cả các cột mong đợi có trong 'input_values_for_ml_sidebar' không.")
            st.stop()
        except Exception as e_general:
            st.error(f"Lỗi không xác định khi xử lý DataFrame đầu vào thô: {e_general}")
            st.stop()

        # Di chuyển debug xuống đây để nó dùng input_df_raw_ml đã được sắp xếp (nếu không lỗi)
        st.subheader("DEBUG: Input cho 7 cột CAT đầu tiên (trước transform)")
        if CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR and not input_df_raw_ml.empty:
            cols_to_debug_sidebar = [col for col in CATEGORICAL_FEATURES_RAW_EXPECTED_BY_PREPROCESSOR[:7] if
                                     col in input_df_raw_ml.columns]
            if cols_to_debug_sidebar:
                st.dataframe(input_df_raw_ml[cols_to_debug_sidebar])
            else:
                st.caption("Không tìm thấy 7 cột CAT đầu tiên để debug trong input_df_raw_ml đã xử lý.")

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
                        f"Lỗi ML: Số tên đặc trưng sau xử lý ({len(FEATURE_NAMES_AFTER_PROCESSING)}) không khớp số cột sau transform ({input_processed_np_array.shape[1]}).")
            except Exception as e_proc:
                st.error(f"Lỗi áp dụng preprocessor: {e_proc}")
                st.error(traceback.format_exc())
        else:
            st.warning("Preprocessor chưa tải hoặc không có dữ liệu ML. Không thể xử lý.")

        current_ml_predictions_ph = {}
        if ml_processing_ok and not processed_df_for_model.empty and ml_models_loaded_global:
            for model_name, model_obj in ml_models_loaded_global.items():
                if model_obj:
                    try:
                        pred_ph = model_obj.predict(processed_df_for_model)
                        current_ml_predictions_ph[model_name] = round(float(pred_ph[0]), 0)
                    except Exception as e_pred_ml:
                        current_ml_predictions_ph[model_name] = "Lỗi"
                else:
                    current_ml_predictions_ph[model_name] = "Mô hình chưa tải"
        else:
            if ml_models_loaded_global:
                for k_model_name in ml_models_loaded_global.keys():
                    current_ml_predictions_ph[k_model_name] = "Lỗi dữ liệu"
        st.session_state.ml_predictions_ph = current_ml_predictions_ph

        kloc_cocomo_ii = calc_loc / 1000
        project_type_for_cocomo_map = {'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached",
                                       'Bảo trì': "Organic", 'Tái cấu trúc': "Semi-detached",
                                       'Tích hợp hệ thống': "Embedded", 'Khác': "Organic"}
        project_type_val_for_cocomo = input_values_for_ml_sidebar.get('Project Type', 'Phát triển mới')
        cocomo_mode_calc = project_type_for_cocomo_map.get(project_type_val_for_cocomo, "Organic")
        effort_pm_cocomo_ii = estimate_cocomo_effort(kloc_cocomo_ii, project_type_cocomo=cocomo_mode_calc)
        st.session_state.cocomo_estimate_ph = round(effort_pm_cocomo_ii * HOURS_PER_PERSON_MONTH, 0)
        st.success("Đã thực hiện dự đoán Effort!")

# --- Khu vực chính hiển thị kết quả ---
main_area_results = st.container()
with main_area_results:
    st.header("🔍 Kết quả Ước tính Chi tiết và Phân tích")

    if st.session_state.raw_input_df_display is not None:
        st.subheader("Dữ liệu đầu vào thô (cho preprocessor ML):")
        st.dataframe(st.session_state.raw_input_df_display, use_container_width=True)

    if st.session_state.processed_input_df_display is not None:
        st.subheader("Dữ liệu đầu vào đã xử lý (sau preprocessor, cho mô hình ML):")
        st.dataframe(st.session_state.processed_input_df_display, use_container_width=True)
        if FEATURE_NAMES_AFTER_PROCESSING:
            st.caption(
                f"Số cột mong đợi (từ feature_names.joblib): {len(FEATURE_NAMES_AFTER_PROCESSING)}. Thực tế: {st.session_state.processed_input_df_display.shape[1]}")

    if st.session_state.ml_predictions_ph:
        st.subheader("📊 Dự đoán Effort từ Mô hình ML (person-hours)")
        ml_preds_to_show = st.session_state.ml_predictions_ph
        ml_model_names = list(ml_preds_to_show.keys())
        num_ml_models_show = len(ml_model_names)
        cols_per_row_show = 3
        for i in range(0, num_ml_models_show, cols_per_row_show):
            row_display_cols = st.columns(cols_per_row_show)
            for j in range(cols_per_row_show):
                if i + j < num_ml_models_show:
                    model_n = ml_model_names[i + j]
                    effort_val = ml_preds_to_show[model_n]
                    with row_display_cols[j]:
                        if isinstance(effort_val, (int, float)):
                            st.metric(label=f"{model_n} (PH)", value=f"{effort_val:,.0f}")
                        else:
                            st.metric(label=f"{model_n} (PH)", value=str(effort_val))
        st.markdown("---")

    if st.session_state.cocomo_estimate_ph is not None:
        st.subheader("⚙️ Ước tính Effort từ COCOMO II (person-hours)")
        st.metric(label="COCOMO II Effort (PH)", value=f"{st.session_state.cocomo_estimate_ph:,.0f}")
        st.markdown("---")

    if st.session_state.ml_predictions_ph or st.session_state.cocomo_estimate_ph is not None:
        st.subheader("📈 Biểu đồ So sánh Effort Tổng hợp (person-hours)")
        chart_data_compare = {}
        if st.session_state.ml_predictions_ph:
            for model_n_chart, effort_val_chart in st.session_state.ml_predictions_ph.items():
                if isinstance(effort_val_chart, (int, float)):
                    chart_data_compare[model_n_chart] = effort_val_chart
        if st.session_state.cocomo_estimate_ph is not None and isinstance(st.session_state.cocomo_estimate_ph,
                                                                          (int, float)):
            chart_data_compare["COCOMO II"] = st.session_state.cocomo_estimate_ph

        if chart_data_compare:
            df_chart = pd.DataFrame(list(chart_data_compare.items()), columns=['Phương pháp', 'Effort (PH)'])
            df_chart = df_chart.sort_values(by='Effort (PH)', ascending=False)
            fig_compare, ax_compare = plt.subplots(figsize=(10, max(6, len(df_chart) * 0.5)))
            bars_compare = ax_compare.bar(df_chart['Phương pháp'], df_chart['Effort (PH)'],
                                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                                 '#e377c2', '#7f7f7f'])
            for bar_item in bars_compare:
                y_val_bar = bar_item.get_height()
                plt.text(bar_item.get_x() + bar_item.get_width() / 2.0,
                         y_val_bar + 0.01 * max(df_chart['Effort (PH)'], default=1),
                         f'{y_val_bar:,.0f}', ha='center', va='bottom', fontsize=9)
            ax_compare.set_ylabel('Effort Ước tính (Person-Hours)', fontsize=12)
            ax_compare.set_title('So sánh Effort Ước tính', fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            ax_compare.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_compare)
        else:
            st.info("Không có đủ dữ liệu hợp lệ để vẽ biểu đồ so sánh.")

    if not st.session_state.ml_predictions_ph and st.session_state.cocomo_estimate_ph is None:
        st.info("Nhập thông tin ở thanh bên trái và nhấn '🚀 Dự đoán Effort (ML & COCOMO II)' để xem kết quả.")

    st.markdown("---")
    st.subheader("📝 Hướng dẫn và Lưu ý Quan Trọng")
    st.markdown(f"""
    1.  **ĐẢM BẢO CÁC FILE `.joblib`:**
        * `preprocessor.joblib`, `feature_names.joblib`, và các file mô hình phải nằm trong thư mục `{OUTPUT_DIR}`.
    2.  **CẤU TRÚC PREPROCESSOR:** Kiểm tra giả định về cấu trúc `preprocessor.joblib` (ColumnTransformer với 'num', 'cat' pipelines, và 'onehot' step).
    3.  **XỬ LÝ CẢNH BÁO "Unknown Categories":**
        * Nguyên nhân chính: Các tùy chọn `selectbox` hoặc giá trị tính toán cho ML không khớp với những gì `OneHotEncoder` đã học.
        * Cách khắc phục: Kiểm tra "DEBUG: Input cho 7 cột CAT đầu tiên" (trong sidebar, sau khi nhấn nút dự đoán) và so sánh với `categorical_options_global` (được trích xuất từ preprocessor).
    4.  **ĐƠN VỊ EFFORT ML:** Code giả định mô hình ML dự đoán Person-Hours.
    5.  **KIỂM TRA SIDEBAR:** Theo dõi thông báo tải tài nguyên.
    """)