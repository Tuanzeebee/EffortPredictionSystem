# Import các thư viện cần thiết
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Để tải mô hình và preprocessors
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import traceback # Để in lỗi chi tiết

# --- Hằng số và Dữ liệu Mô phỏng (Giữ lại từ code gốc) ---
COCOMO_A = 2.4
COCOMO_B = 1.05
COCOMO_C = 2.5
COCOMO_D = 0.38
EFFORT_PER_UCP = 20  # Person-hours per UCP
HOURS_PER_PERSON_MONTH = 152  # Số giờ làm việc trung bình mỗi tháng cho một người

# Dữ liệu này vẫn cần thiết cho việc tính toán sơ bộ và quy đổi LOC/FP
AVG_LOC_PER_FP = {
    'Java': 53, 'Python': 35, 'C++': 47, 'C#': 54, 'JavaScript': 47,
    'SQL': 15, 'COBOL': 90, 'ABAP': 70, 'PHP': 40, 'Swift': 30,
    'Kotlin': 32, 'Ruby': 25, 'Go': 45, 'Assembly': 200,
    'Scripting': 20, 'Visual Basic': 32, 'Ada': 71, 'Perl': 27,
    'Khác': 50
}

# --- Định nghĩa đường dẫn cho Mô hình và Preprocessor (Từ code thứ hai) ---
OUTPUT_DIR = "." # Giả sử các file .joblib nằm cùng thư mục với script
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.joblib") # Tên các cột SAU KHI preprocessor xử lý
MODEL_PATHS = OrderedDict([
    ('Linear Regression', os.path.join(OUTPUT_DIR, "linear_regression_model.joblib")),
    ('Decision Tree', os.path.join(OUTPUT_DIR, "decision_tree_model.joblib")),
    ('Random Forest', os.path.join(OUTPUT_DIR, "random_forest_model.joblib")),
    ('XGBoost', os.path.join(OUTPUT_DIR, "xgboost_model.joblib")), # Đổi tên file cho khớp
    ('MLP Regressor', os.path.join(OUTPUT_DIR, "mlp_regressor_model.joblib"))
])

# --- Các lựa chọn cho Selectbox (Sẽ được cập nhật động từ preprocessor nếu có thể) ---
# Khởi tạo rỗng, sẽ được điền bởi load_artifacts_updated
PROJECT_TYPES_OPTIONS = ['Phát triển mới', 'Nâng cấp lớn', 'Bảo trì', 'Khác']
LANGUAGE_TYPES_OPTIONS = ['3GL', '4GL', 'Khác']
PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS = sorted(list(AVG_LOC_PER_FP.keys())) # Vẫn giữ cho quy đổi LOC/FP
COUNT_APPROACH_OPTIONS = ['IFPUG', 'NESMA', 'Khác']
APPLICATION_GROUP_OPTIONS = ['Nghiệp vụ (Business)', 'Hỗ trợ Quyết định (Decision Support)', 'Khác']
APPLICATION_TYPES_OPTIONS = ['Ứng dụng Web', 'Ứng dụng Di động', 'Khác']
DEVELOPMENT_TYPES_OPTIONS = ['Nội bộ (In-house)', 'Thuê ngoài (Outsource)', 'Khác']

# Các cột này sẽ được xác định từ preprocessor (original_cols_order_loaded)
# Chúng là các cột đầu vào THÔ mà preprocessor mong đợi.
NUMERICAL_FEATURES_RAW_EXPECTED = [] # Sẽ được cập nhật
CATEGORICAL_FEATURES_RAW_EXPECTED = [] # Sẽ được cập nhật
ORIGINAL_COLS_ORDER_LOADED = [] # Thứ tự các cột đầu vào thô mà preprocessor mong đợi

# Danh sách này sẽ là tên các cột SAU KHI preprocessor xử lý (tải từ FEATURES_PATH)
# Nó thay thế cho X_TRAIN_COLUMNS_ORDERED được xây dựng thủ công.
FEATURE_NAMES_AFTER_PROCESSING_LOADED = []


# --- Hàm Tính Toán (từ code gốc, giữ nguyên) ---
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
        if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio
        kloc = calculated_loc / 1000
        if kloc > 0:
            effort_pm_from_loc = COCOMO_A * (kloc ** COCOMO_B)
            if EFFORT_PER_UCP > 0 and HOURS_PER_PERSON_MONTH > 0:
                 # Ước tính UCP từ Effort (PM) quy đổi từ LOC, sau đó UCP * EFFORT_PER_UCP = Effort (PH)
                 # Effort (PH) = effort_pm_from_loc * HOURS_PER_PERSON_MONTH
                 # calculated_ucp = (effort_pm_from_loc * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
                 # Cách tính UCP ở đây cần xem lại nếu muốn nó độc lập hơn
                 # Hiện tại, giữ nguyên logic cũ của code gốc
                 pass # Logic UCP từ LOC/FP trong code gốc hơi vòng, tạm bỏ qua để tránh phức tạp hóa
    elif size_metric_choice == 'FP':
        calculated_fp = size_metric_value
        calculated_loc = calculated_fp * loc_fp_ratio
        # kloc = calculated_loc / 1000 # Phần này sẽ được tính ở dưới
    elif size_metric_choice == 'UCP':
        calculated_ucp = size_metric_value
        # Nếu UCP là đầu vào, tính effort từ UCP trước, rồi quy ra LOC/FP
        if EFFORT_PER_UCP > 0 and HOURS_PER_PERSON_MONTH > 0:
            effort_ph_from_ucp = calculated_ucp * EFFORT_PER_UCP
            effort_pm_from_ucp = effort_ph_from_ucp / HOURS_PER_PERSON_MONTH
            if COCOMO_A > 0 and COCOMO_B != 0 and effort_pm_from_ucp > 0:
                base_cocomo_val = effort_pm_from_ucp / COCOMO_A
                if base_cocomo_val > 0:
                    kloc_from_ucp_effort = base_cocomo_val ** (1 / COCOMO_B)
                    calculated_loc = kloc_from_ucp_effort * 1000
                    if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio


    # Tính toán lại LOC, FP, UCP dựa trên đầu vào chính và các quy đổi
    if size_metric_choice == 'LOC':
        # calculated_loc đã có
        if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio
        # Để tính UCP từ LOC: LOC -> KLOC -> Effort (COCOMO PM) -> Effort (PH) -> UCP
        _kloc = calculated_loc / 1000
        if _kloc > 0:
            _effort_pm = COCOMO_A * (_kloc ** COCOMO_B)
            if EFFORT_PER_UCP > 0:
                 calculated_ucp = (_effort_pm * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP

    elif size_metric_choice == 'FP':
        # calculated_fp đã có
        calculated_loc = calculated_fp * loc_fp_ratio
        _kloc = calculated_loc / 1000
        if _kloc > 0:
            _effort_pm = COCOMO_A * (_kloc ** COCOMO_B)
            if EFFORT_PER_UCP > 0:
                calculated_ucp = (_effort_pm * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
    # Trường hợp UCP là đầu vào đã xử lý ở trên để ra calculated_loc, calculated_fp

    final_kloc = calculated_loc / 1000
    if final_kloc > 0:
        estimated_effort_pm = COCOMO_A * (final_kloc ** COCOMO_B)
        if estimated_effort_pm > 0:
            # COCOMO C và D dùng để tính Development Time từ Effort
            # Effort^(COCOMO_D)
            # Duration = C * (Effort)^D
            dev_time_base_for_formula = estimated_effort_pm # Effort ở đây là PM
            if dev_time_base_for_formula > 0:
                 estimated_dev_time_months = COCOMO_C * (dev_time_base_for_formula ** COCOMO_D)
            if estimated_dev_time_months > 0:
                estimated_team_size = estimated_effort_pm / estimated_dev_time_months
            else:
                estimated_team_size = 1 if estimated_effort_pm > 0 else 0
    else: # Xử lý trường hợp final_kloc = 0
        estimated_effort_pm = 0
        estimated_dev_time_months = 0
        estimated_team_size = 0


    return (
        round(calculated_loc, 2), round(calculated_fp, 2), round(calculated_ucp, 2),
        round(estimated_effort_pm, 2), round(estimated_dev_time_months, 2), round(estimated_team_size, 2)
    )

# --- Hàm COCOMO II (Giữ nguyên từ code gốc) ---
def estimate_cocomo_effort(kloc, project_type_cocomo="Organic", cost_drivers=None):
    effort_multipliers = 1.0
    if cost_drivers:
        for driver_value in cost_drivers.values():
            effort_multipliers *= driver_value

    cocomo_params = {
        "Organic": (2.4, 1.05), # a, b
        "Semi-detached": (3.0, 1.12),
        "Embedded": (3.6, 1.20)
    }
    # Thêm các tham số cho Development Time (c, d) và Staffing (e, f) nếu cần COCOMO chi tiết hơn
    # Ví dụ: params_time = {"Organic": (2.5, 0.38), ...}

    a, b = cocomo_params.get(project_type_cocomo, cocomo_params["Organic"])

    if kloc <= 0: return 0.0
    effort_pm = a * (kloc ** b) * effort_multipliers
    return round(effort_pm, 2)


# --- Hàm tải mô hình và preprocessors (CẬP NHẬT DỰA TRÊN CODE THỨ HAI) ---
@st.cache_resource # Cache để không tải lại mỗi lần tương tác
def load_artifacts_updated():
    global NUMERICAL_FEATURES_RAW_EXPECTED, CATEGORICAL_FEATURES_RAW_EXPECTED
    global PROJECT_TYPES_OPTIONS, LANGUAGE_TYPES_OPTIONS, COUNT_APPROACH_OPTIONS
    global APPLICATION_GROUP_OPTIONS, APPLICATION_TYPES_OPTIONS, DEVELOPMENT_TYPES_OPTIONS
    global ORIGINAL_COLS_ORDER_LOADED, FEATURE_NAMES_AFTER_PROCESSING_LOADED

    loaded_preprocessor = None
    loaded_feature_names = []
    loaded_models = OrderedDict()
    original_cols_order = []
    categorical_features_options = {}
    all_loaded_successfully = True

    # --- Tải Preprocessor ---
    if not os.path.exists(PREPROCESSOR_PATH):
        st.sidebar.error(f"LỖI: Không tìm thấy file preprocessor tại '{PREPROCESSOR_PATH}'")
        all_loaded_successfully = False
    else:
        try:
            loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
            st.sidebar.success("Preprocessor đã được tải.")
            # --- Trích xuất thông tin từ Preprocessor ---
            try:
                # Giả định preprocessor là ColumnTransformer
                # và có transformers tên là 'num' và 'cat'
                num_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'num')
                cat_transformer_tuple = next(t for t in loaded_preprocessor.transformers_ if t[0] == 'cat')

                original_num_features = list(num_transformer_tuple[2])
                original_cat_features = list(cat_transformer_tuple[2])
                original_cols_order = original_num_features + original_cat_features
                ORIGINAL_COLS_ORDER_LOADED.extend(original_cols_order) # Cập nhật biến global
                NUMERICAL_FEATURES_RAW_EXPECTED.extend(original_num_features)
                CATEGORICAL_FEATURES_RAW_EXPECTED.extend(original_cat_features)

                st.sidebar.info(f"Preprocessor mong đợi các cột số: {original_num_features}")
                st.sidebar.info(f"Preprocessor mong đợi các cột loại: {original_cat_features}")


                # Trích xuất các categories từ OneHotEncoder bên trong pipeline của 'cat'
                cat_pipeline = loaded_preprocessor.named_transformers_['cat']
                onehot_encoder = cat_pipeline.named_steps['onehot'] # Giả sử bước onehot tên là 'onehot'

                if hasattr(onehot_encoder, 'categories_'):
                    if len(onehot_encoder.categories_) == len(original_cat_features):
                        for i, feature_name in enumerate(original_cat_features):
                            categories = onehot_encoder.categories_[i].tolist()
                            categorical_features_options[feature_name] = categories
                            # Cập nhật các list OPTIONS dựa trên feature_name
                            if feature_name == 'Project Type': PROJECT_TYPES_OPTIONS = categories
                            elif feature_name == 'Language Type': LANGUAGE_TYPES_OPTIONS = categories
                            # Thêm các elif khác cho các cột phân loại còn lại
                            # Ví dụ:
                            # elif feature_name == 'Primary Programming Language': PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS_FROM_PREP = categories # Cẩn thận nếu cột này cũng có trong preprocessor
                            elif feature_name == 'Count Approach': COUNT_APPROACH_OPTIONS = categories
                            elif feature_name == 'Application Group': APPLICATION_GROUP_OPTIONS = categories
                            elif feature_name == 'Application Type': APPLICATION_TYPES_OPTIONS = categories
                            elif feature_name == 'Development Type': DEVELOPMENT_TYPES_OPTIONS = categories

                        st.sidebar.success("Các tùy chọn cho trường phân loại đã được cập nhật từ preprocessor.")
                    else:
                        st.sidebar.error(f"Lỗi trích xuất: Số lượng categories ({len(onehot_encoder.categories_)}) không khớp số cột loại ({len(original_cat_features)}).")
                        all_loaded_successfully = False
                else:
                    st.sidebar.error("Lỗi trích xuất: Không tìm thấy 'categories_' trong OneHotEncoder.")
                    all_loaded_successfully = False
            except Exception as e_extract:
                st.sidebar.error(f"Lỗi khi trích xuất thông tin từ preprocessor: {e_extract}")
                all_loaded_successfully = False
        except Exception as e_load_prep:
            st.sidebar.error(f"Lỗi nghiêm trọng khi tải preprocessor: {e_load_prep}")
            all_loaded_successfully = False

    # --- Tải Feature Names (sau khi xử lý) ---
    if not os.path.exists(FEATURES_PATH):
        st.sidebar.error(f"LỖI: Không tìm thấy file tên đặc trưng (features) tại '{FEATURES_PATH}'")
        all_loaded_successfully = False
    else:
        try:
            loaded_feature_names = joblib.load(FEATURES_PATH)
            if isinstance(loaded_feature_names, np.ndarray): loaded_feature_names = loaded_feature_names.tolist()
            if not isinstance(loaded_feature_names, list):
                 loaded_feature_names = list(loaded_feature_names) # Cố gắng chuyển đổi
            FEATURE_NAMES_AFTER_PROCESSING_LOADED.extend(loaded_feature_names) # Cập nhật biến global
            st.sidebar.success(f"Tên các đặc trưng sau xử lý ({len(loaded_feature_names)} cột) đã được tải.")
        except Exception as e_load_feat:
            st.sidebar.error(f"Lỗi khi tải feature names: {e_load_feat}")
            all_loaded_successfully = False

    # --- Tải các Mô hình ML ---
    models_actually_loaded = 0
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            st.sidebar.warning(f"Cảnh báo: Không tìm thấy file mô hình '{name}' tại '{path}'. Bỏ qua.")
            loaded_models[name] = None # Đánh dấu là không tải được
            continue
        try:
            model = joblib.load(path)
            loaded_models[name] = model
            models_actually_loaded += 1
            # st.sidebar.info(f"Tải mô hình ML {name} thành công.")
        except Exception as e_load_model:
            st.sidebar.warning(f"Lỗi khi tải mô hình {name}: {e_load_model}. Đặt là None.")
            loaded_models[name] = None

    if models_actually_loaded > 0:
        st.sidebar.success(f"Đã tải thành công {models_actually_loaded}/{len(MODEL_PATHS)} mô hình ML.")
    else:
        st.sidebar.error("LỖI: Không tải được bất kỳ mô hình Machine Learning nào.")
        all_loaded_successfully = False


    if not all_loaded_successfully:
        st.sidebar.error("Có lỗi xảy ra trong quá trình tải một hoặc nhiều tài nguyên ML.")
        # Trả về những gì đã tải được để có thể debug hoặc dùng phần khác
        return loaded_preprocessor, loaded_feature_names, loaded_models, original_cols_order, categorical_features_options, False

    return loaded_preprocessor, loaded_feature_names, loaded_models, original_cols_order, categorical_features_options, True

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v4 (Tích hợp Preprocessor)")

# Tải các tài nguyên ML một lần
(   preprocessor_loaded,
    feature_names_loaded, # Đây sẽ là X_TRAIN_COLUMNS_ORDERED của chúng ta
    ml_models_loaded,
    original_cols_for_ml_input, # Các cột thô mà preprocessor mong đợi
    categorical_options_from_ml, # Các options cho trường phân loại từ preprocessor
    load_successful
) = load_artifacts_updated()


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
        "Chọn chỉ số kích thước đầu vào chính:", ('LOC', 'FP', 'UCP'), key='size_metric_choice_v4'
    )
    size_metric_label = f"Nhập giá trị cho {size_metric_choice}:"
    default_val_size, step_val_size = (10000.0, 1000.0) if size_metric_choice == 'LOC' else \
                                     (200.0, 10.0) if size_metric_choice == 'FP' else \
                                     (100.0, 5.0)
    size_metric_value = st.number_input(
        size_metric_label, min_value=0.0, value=default_val_size, step=step_val_size, key='size_metric_value_v4', format="%.2f"
    )
    selected_primary_programming_language_for_conversion = st.selectbox(
        "Ngôn ngữ chính (cho quy đổi LOC/FP/UCP):", PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS,
        index=PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS.index('Java') if 'Java' in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS else 0,
        key='selected_primary_programming_language_for_conversion_v4'
    )

    (calculated_loc, calculated_fp, calculated_ucp,
     estimated_effort_pm_cocomo_basic,
     estimated_dev_time_months_cocomo_basic,
     estimated_team_size_cocomo_basic) = calculate_metrics(
        size_metric_choice, size_metric_value, selected_primary_programming_language_for_conversion
    )

    st.markdown("---")
    st.subheader("📈 Các Chỉ số Kích thước Ước tính:")
    col_loc_ucp, col_fp_empty = st.columns(2)
    with col_loc_ucp:
        st.metric(label="LOC", value=f"{calculated_loc:,.0f}", delta="Tính toán" if size_metric_choice != 'LOC' else "Đầu vào", delta_color="off")
        st.metric(label="UCP", value=f"{calculated_ucp:,.0f}", delta="Tính toán" if size_metric_choice != 'UCP' else "Đầu vào", delta_color="off")
    with col_fp_empty:
        st.metric(label="FP", value=f"{calculated_fp:,.0f}", delta="Tính toán" if size_metric_choice != 'FP' else "Đầu vào", delta_color="off")

    st.markdown("---")
    st.subheader("⏱️ Ước tính Sơ bộ (COCOMO cơ bản):")
    col_e_pm, col_t_m, col_s_p = st.columns(3)
    with col_e_pm:
        st.metric(label="Nỗ lực (Person-Months)", value=f"{estimated_effort_pm_cocomo_basic:,.1f}")
    with col_t_m:
        st.metric(label="T.Gian P.Triển (Tháng)", value=f"{estimated_dev_time_months_cocomo_basic:,.1f}")
    with col_s_p:
        st.metric(label="Quy mô Nhóm (Người)", value=f"{estimated_team_size_cocomo_basic:,.1f}")

    st.markdown("---")
    st.subheader("📋 Thông tin Chi tiết Dự án (cho Model ML):")

    input_values_for_ml = {} # Thu thập các giá trị cho ML

    # Sử dụng original_cols_for_ml_input và categorical_options_from_ml để tạo input
    if load_successful and preprocessor_loaded and original_cols_for_ml_input:
        st.markdown("_Các trường dưới đây được yêu cầu bởi mô hình ML đã tải._")
        # Lấy các giá trị số đã tính toán nếu chúng nằm trong các cột ML yêu cầu
        if 'LOC' in original_cols_for_ml_input:
            input_values_for_ml['LOC'] = calculated_loc
            st.text(f"LOC (từ tính toán trên): {calculated_loc:,.0f}")
        if 'FP' in original_cols_for_ml_input:
            input_values_for_ml['FP'] = calculated_fp
            st.text(f"FP (từ tính toán trên): {calculated_fp:,.0f}")
        if 'UCP' in original_cols_for_ml_input:
            input_values_for_ml['UCP'] = calculated_ucp
            st.text(f"UCP (từ tính toán trên): {calculated_ucp:,.0f}")
        if 'Development Time (months)' in original_cols_for_ml_input:
            input_values_for_ml['Development Time (months)'] = estimated_dev_time_months_cocomo_basic
            st.text(f"Development Time (months) (từ COCOMO Basic): {estimated_dev_time_months_cocomo_basic:,.1f}")
        if 'Team Size' in original_cols_for_ml_input:
            input_values_for_ml['Team Size'] = estimated_team_size_cocomo_basic
            st.text(f"Team Size (từ COCOMO Basic): {estimated_team_size_cocomo_basic:,.1f}")

        # Các trường số khác nếu có (ví dụ, người dùng tự nhập)
        for feature_name in NUMERICAL_FEATURES_RAW_EXPECTED:
            if feature_name not in ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']:
                 # Cho phép người dùng nhập nếu mô hình cần thêm trường số
                 input_values_for_ml[feature_name] = st.number_input(
                     f"{feature_name} (cho Model ML):",
                     value=0.0, format="%.2f", key=f"ml_num_{feature_name}"
                 )

        # Các trường phân loại
        for feature_name in CATEGORICAL_FEATURES_RAW_EXPECTED:
            options = categorical_options_from_ml.get(feature_name, ['Vui lòng kiểm tra preprocessor'])
            # Tìm index cho ngôn ngữ lập trình nếu có
            default_idx = 0
            if feature_name == 'Primary Programming Language' and selected_primary_programming_language_for_conversion in options:
                default_idx = options.index(selected_primary_programming_language_for_conversion)

            input_values_for_ml[feature_name] = st.selectbox(
                f"{feature_name} (cho Model ML):", options, index=default_idx, key=f"ml_cat_{feature_name}"
            )
    else:
        st.warning("Không thể tạo các trường nhập liệu chi tiết cho ML do lỗi tải tài nguyên.")
        # Fallback: Hiển thị các selectbox với options mặc định từ code gốc cũ (ít linh hoạt hơn)
        input_values_for_ml['Project Type'] = st.selectbox("Project Type:", PROJECT_TYPES_OPTIONS, key='input_project_type_v4_fb')
        input_values_for_ml['Language Type'] = st.selectbox("Language Type:", LANGUAGE_TYPES_OPTIONS, key='input_language_type_v4_fb')
        input_values_for_ml['Primary Programming Language'] = st.selectbox("Primary Programming Language (cho Model ML):",
                                                                          PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS,
                                                                          index=PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS.index(selected_primary_programming_language_for_conversion) if selected_primary_programming_language_for_conversion in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS else 0,
                                                                          key='input_primary_lang_model_v4_fb')
        # ... thêm các fallback khác nếu cần

    st.markdown("---")
    if st.button("🚀 Dự đoán Effort Chính (ML & COCOMO II)", key='predict_effort_button_v4', disabled=not load_successful):
        # 1. Thu thập dữ liệu đầu vào cho DataFrame thô
        # (input_values_for_ml đã được thu thập ở trên)
        # Đảm bảo tất cả các cột mà preprocessor mong đợi đều có mặt
        input_data_dict_final = {}
        if original_cols_for_ml_input: # Nếu đã tải được original_cols_for_ml_input
            for col_name in original_cols_for_ml_input:
                input_data_dict_final[col_name] = input_values_for_ml.get(col_name) # Lấy giá trị, nếu thiếu sẽ là None (preprocessor sẽ xử lý)
        else: # Fallback rất cơ bản nếu không tải được original_cols_for_ml_input
             # Điều này không nên xảy ra nếu disabled=not load_successful hoạt động đúng
             st.error("Không có thông tin về các cột đầu vào cho preprocessor.")
             input_data_dict_final = input_values_for_ml # Lấy tất cả những gì có


        input_df_raw_for_ml = pd.DataFrame([input_data_dict_final])
        # Sắp xếp lại các cột của DataFrame thô theo đúng thứ tự mà preprocessor mong đợi
        if original_cols_for_ml_input:
            try:
                input_df_raw_for_ml = input_df_raw_for_ml[original_cols_for_ml_input]
                st.session_state.raw_input_df_display = input_df_raw_for_ml.copy()
            except KeyError as e:
                st.error(f"Lỗi sắp xếp cột cho preprocessor: {e}. Thiếu cột trong input_values_for_ml.")
                st.stop() # Dừng xử lý nếu thiếu cột quan trọng
        else: # Nếu không có original_cols_for_ml_input
            st.session_state.raw_input_df_display = input_df_raw_for_ml.copy() # Hiển thị những gì có


        # 2. Tiền xử lý input DataFrame bằng preprocessor đã tải
        input_df_final_for_model = pd.DataFrame() # Khởi tạo
        processed_successfully_ml = False
        if preprocessor_loaded and not input_df_raw_for_ml.empty:
            try:
                st.write("Dữ liệu đầu vào thô cho preprocessor:", input_df_raw_for_ml)
                input_processed_np = preprocessor_loaded.transform(input_df_raw_for_ml)

                if feature_names_loaded and len(feature_names_loaded) == input_processed_np.shape[1]:
                    input_df_final_for_model = pd.DataFrame(input_processed_np, columns=feature_names_loaded)
                    st.session_state.processed_input_df_display = input_df_final_for_model.copy()
                    processed_successfully_ml = True
                    st.success("Dữ liệu đã được tiền xử lý thành công bởi preprocessor.")
                else:
                    st.error(f"Lỗi ML: Số lượng tên đặc trưng sau xử lý ({len(feature_names_loaded)}) không khớp số cột sau transform ({input_processed_np.shape[1]}).")
            except Exception as e_process:
                st.error(f"Lỗi khi áp dụng preprocessor: {e_process}")
                st.error(traceback.format_exc())
        else:
            st.warning("Preprocessor chưa được tải hoặc không có dữ liệu đầu vào. Không thể xử lý cho ML.")

        # 3. Thực hiện dự đoán ML (Giả định mô hình trả về Person-Hours trực tiếp)
        ml_predictions_ph_current = {}
        if processed_successfully_ml and not input_df_final_for_model.empty and ml_models_loaded:
            for model_name, model_object in ml_models_loaded.items():
                if model_object is not None:
                    try:
                        prediction_ph = model_object.predict(input_df_final_for_model)
                        # Giả định prediction_ph[0] là giá trị effort bằng Person-Hours
                        ml_predictions_ph_current[model_name] = round(float(prediction_ph[0]), 0)
                    except Exception as e_pred:
                        st.error(f"Lỗi khi dự đoán với mô hình {model_name}: {e_pred}")
                        ml_predictions_ph_current[model_name] = "Lỗi dự đoán"
                else:
                    ml_predictions_ph_current[model_name] = "Mô hình chưa tải"
        else:
            st.warning("Không thể thực hiện dự đoán ML do lỗi chuẩn bị dữ liệu hoặc thiếu mô hình.")
            if ml_models_loaded: # Nếu có danh sách model
                for model_name_key in ml_models_loaded.keys(): ml_predictions_ph_current[model_name_key] = "Lỗi dữ liệu"

        st.session_state.ml_predictions_ph = ml_predictions_ph_current

        # 4. Tính toán COCOMO II (Effort là Person-Months, sau đó chuyển sang Person-Hours)
        kloc_for_cocomo = calculated_loc / 1000
        cocomo_project_type_map = {
            'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached",
            'Bảo trì': "Organic", 'Tái cấu trúc': "Semi-detached",
            'Tích hợp hệ thống': "Embedded", 'Khác': "Organic"
        }
        # Lấy project type từ input cho ML nếu có, nếu không thì dùng giá trị mặc định
        project_type_for_cocomo_input = input_values_for_ml.get('Project Type', 'Phát triển mới')

        cocomo_type_for_calc = cocomo_project_type_map.get(project_type_for_cocomo_input, "Organic")
        cocomo_effort_pm_estimated = estimate_cocomo_effort(kloc_for_cocomo, project_type_cocomo=cocomo_type_for_calc)
        st.session_state.cocomo_estimate_ph = round(cocomo_effort_pm_estimated * HOURS_PER_PERSON_MONTH, 0)

        st.success("Đã thực hiện dự đoán Effort!")

# --- Khu vực chính ---
main_area = st.container()
with main_area:
    st.header("🔍 Kết quả Ước tính Chi tiết và Phân tích")

    if st.session_state.raw_input_df_display is not None:
        st.subheader("Dữ liệu đầu vào thô (trước khi qua preprocessor ML):")
        st.dataframe(st.session_state.raw_input_df_display, use_container_width=True)

    if st.session_state.processed_input_df_display is not None:
        st.subheader("Dữ liệu đầu vào đã xử lý (sau preprocessor, cho mô hình ML):")
        st.dataframe(st.session_state.processed_input_df_display, use_container_width=True)
        if feature_names_loaded:
            st.caption(
                f"Số cột mong đợi bởi mô hình (từ feature_names.joblib): {len(feature_names_loaded)}. "
                f"Số cột thực tế truyền vào: {st.session_state.processed_input_df_display.shape[1]}"
            )

    if st.session_state.ml_predictions_ph:
        st.subheader("📊 Dự đoán Effort Chính từ các Mô hình ML (person-hours)")
        current_ml_predictions_ph = st.session_state.ml_predictions_ph
        model_names_ml = list(current_ml_predictions_ph.keys())
        num_models_ml = len(model_names_ml)
        cols_per_row = 3
        for i in range(0, num_models_ml, cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_models_ml:
                    model_name = model_names_ml[i + j]
                    effort_value_ph = current_ml_predictions_ph[model_name]
                    with row_cols[j]:
                        if isinstance(effort_value_ph, (int, float)):
                            st.metric(label=f"{model_name} (PH)", value=f"{effort_value_ph:,.0f}")
                        else:
                            st.metric(label=f"{model_name} (PH)", value=str(effort_value_ph))
        st.markdown("---")

        st.subheader("📈 Biểu đồ So sánh Effort Tổng hợp (person-hours)")
        comparison_data_for_chart = {}
        for model_name_chart, effort_ph_chart in current_ml_predictions_ph.items():
            if isinstance(effort_ph_chart, (int, float)):
                comparison_data_for_chart[model_name_chart] = effort_ph_chart

        if st.session_state.cocomo_estimate_ph is not None and isinstance(st.session_state.cocomo_estimate_ph, (int, float)):
            comparison_data_for_chart["COCOMO II"] = st.session_state.cocomo_estimate_ph

        if comparison_data_for_chart:
            df_comparison_chart = pd.DataFrame(list(comparison_data_for_chart.items()), columns=['Phương pháp', 'Effort (Person-Hours)'])
            df_comparison_chart = df_comparison_chart.sort_values(by='Effort (Person-Hours)', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 7))
            bars = ax.bar(df_comparison_chart['Phương pháp'], df_comparison_chart['Effort (Person-Hours)'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02 * max(df_comparison_chart['Effort (Person-Hours)'], default=1), f'{yval:,.0f}', ha='center', va='bottom', fontsize=9) # Thêm default cho max
            ax.set_ylabel('Effort Ước tính (Person-Hours)', fontsize=12)
            ax.set_title('So sánh Effort Ước tính giữa các Phương pháp', fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Không có đủ dữ liệu hợp lệ để vẽ biểu đồ so sánh tổng hợp.")

    if st.session_state.cocomo_estimate_ph is not None:
        st.subheader("⚙️ Ước tính Effort từ COCOMO II (person-hours) - Chi tiết")
        st.metric(label="COCOMO II Effort (PH)", value=f"{st.session_state.cocomo_estimate_ph:,.0f}")

    if not st.session_state.ml_predictions_ph and st.session_state.cocomo_estimate_ph is None:
        st.info("Nhập thông tin ở thanh bên trái và nhấn '🚀 Dự đoán Effort Chính (ML & COCOMO II)' để xem kết quả.")

    st.markdown("---")
    st.subheader("📝 Chỉ số Đánh giá Mô hình (Tùy chọn)")
    st.info("Để hiển thị các chỉ số đánh giá của các mô hình ML, bạn cần tải chúng từ quá trình huấn luyện và hiển thị tại đây.")

    st.markdown("---")
    st.subheader("Hướng dẫn Tiếp theo và Lưu ý Quan Trọng")
    st.markdown(f"""
    1.  **ĐẢM BẢO CÁC FILE `.joblib`:**
        * `preprocessor.joblib`: Chứa ColumnTransformer đã được huấn luyện.
        * `feature_names.joblib`: Chứa danh sách tên các cột SAU KHI dữ liệu đã qua preprocessor.
        * Các file mô hình: `linear_regression_model.joblib`, `decision_tree_model.joblib`, v.v.
        * Tất cả các file này cần nằm trong thư mục `{OUTPUT_DIR}` (hiện tại là thư mục chứa script này).
    2.  **CẤU TRÚC PREPROCESSOR:** Mã này giả định `preprocessor.joblib` là một `ColumnTransformer` của scikit-learn, và nó chứa các bước xử lý có tên là `'num'` cho các cột số và `'cat'` cho các cột phân loại. Bên trong pipeline `'cat'`, cần có một bước OneHotEncoder tên là `'onehot'`. Nếu cấu trúc của bạn khác, bạn cần điều chỉnh phần trích xuất thông tin trong hàm `load_artifacts_updated()`.
    3.  **ĐƠN VỊ EFFORT CỦA MÔ HÌNH ML:** Mã này hiện **giả định các mô hình ML đã được huấn luyện để dự đoán effort trực tiếp bằng đơn vị Person-Hours**. Nếu mô hình của bạn dự đoán Person-Months, bạn cần điều chỉnh lại logic (ví dụ, nhân kết quả dự đoán ML với `HOURS_PER_PERSON_MONTH`).
    4.  **THỨ TỰ CỘT ĐẦU VÀO CHO ML:** Preprocessor sẽ xử lý các cột đầu vào theo thứ tự mà nó đã học (`original_cols_for_ml_input`). Đảm bảo rằng dữ liệu bạn cung cấp cho ML (từ các tính toán sơ bộ và nhập liệu của người dùng) được sắp xếp đúng thứ tự này trước khi đưa vào `preprocessor.transform()`.
    5.  **KIỂM TRA LỖI TẢI Ở SIDEBAR:** Các thông báo lỗi/thành công khi tải preprocessor, features, và models sẽ xuất hiện ở sidebar. Hãy kiểm tra kỹ nếu có vấn đề.
    """)

# Để chạy ứng dụng này:
# 1. Cài đặt: streamlit, pandas, numpy, scikit-learn, joblib, matplotlib, (xgboost nếu dùng)
# 2. Chuẩn bị các file .joblib (preprocessor.joblib, feature_names.joblib, các file model) và đặt chúng vào cùng thư mục với file script này (hoặc cập nhật OUTPUT_DIR).
# 3. Chạy lệnh: streamlit run your_script_name.py