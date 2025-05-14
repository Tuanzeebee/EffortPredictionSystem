# Import các thư viện cần thiết
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Để tải mô hình và preprocessors
import matplotlib.pyplot as plt  # Thêm thư viện matplotlib

# --- Hằng số và Dữ liệu Mô phỏng ---
COCOMO_A = 2.4
COCOMO_B = 1.05
COCOMO_C = 2.5
COCOMO_D = 0.38
EFFORT_PER_UCP = 20
HOURS_PER_PERSON_MONTH = 152  # Số giờ làm việc trung bình mỗi tháng cho một người

# Dữ liệu này vẫn cần thiết cho việc tính toán sơ bộ và quy đổi LOC/FP
AVG_LOC_PER_FP = {
    'Java': 53, 'Python': 35, 'C++': 47, 'C#': 54, 'JavaScript': 47,
    'SQL': 15, 'COBOL': 90, 'ABAP': 70, 'PHP': 40, 'Swift': 30,
    'Kotlin': 32, 'Ruby': 25, 'Go': 45, 'Assembly': 200,
    'Scripting': 20, 'Visual Basic': 32, 'Ada': 71, 'Perl': 27,
    'Khác': 50  # Giá trị mặc định/trung bình
}

# --- Cập nhật các lựa chọn cho Selectbox dựa trên cột mới ---
# Các giá trị này bạn nên cập nhật từ dữ liệu thực tế của mình
PROJECT_TYPES_OPTIONS = ['Phát triển mới', 'Nâng cấp lớn', 'Bảo trì', 'Tái cấu trúc', 'Tích hợp hệ thống',
                         'Khác']  # Giữ nguyên hoặc cập nhật
LANGUAGE_TYPES_OPTIONS = ['3GL', '4GL', 'Assembly', 'Scripting', 'Ngôn ngữ truy vấn (SQL)',
                          'Ngôn ngữ đánh dấu (HTML/XML)', 'Khác']  # Giữ nguyên hoặc cập nhật
PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS = sorted(
    list(AVG_LOC_PER_FP.keys()))  # Lấy từ AVG_LOC_PER_FP hoặc danh sách thực tế
COUNT_APPROACH_OPTIONS = ['IFPUG', 'NESMA', 'FiSMA', 'COSMIC', 'Mark II', 'LOC Manual', 'Khác']  # Ví dụ
APPLICATION_GROUP_OPTIONS = ['Nghiệp vụ (Business)', 'Hỗ trợ Quyết định (Decision Support)',
                             'Khoa học/Kỹ thuật (Scientific/Engineering)', 'Thời gian thực (Real-time)',
                             'Hệ thống (System Software)', 'Tiện ích (Utility)', 'Khác']  # Ví dụ
APPLICATION_TYPES_OPTIONS = ['Ứng dụng Web', 'Ứng dụng Di động', 'Ứng dụng Desktop', 'Hệ thống Nhúng',
                             'Xử lý Dữ liệu/Batch', 'API/Dịch vụ', 'Trí tuệ nhân tạo/ML', 'Game',
                             'Khác']  # Giữ nguyên hoặc cập nhật
DEVELOPMENT_TYPES_OPTIONS = ['Nội bộ (In-house)', 'Thuê ngoài (Outsource)', 'Hỗn hợp (Hybrid)',
                             'Mã nguồn mở (Đóng góp)', 'Sản phẩm (COTS) tùy chỉnh', 'Khác']  # Giữ nguyên hoặc cập nhật

# --- Định nghĩa các cột (quan trọng cho tiền xử lý) ---
# Cột mục tiêu (target variable) thường là 'Effort (person-hours)' và không nằm trong features
NUMERICAL_FEATURES_RAW = ['LOC', 'FP', 'UCP', 'Development Time (months)', 'Team Size']

CATEGORICAL_FEATURES_RAW = [
    'Project Type',
    'Language Type',
    'Primary Programming Language',
    'Count Approach',
    'Application Group',
    'Application Type',
    'Development Type'
]

# ----- !!! QUAN TRỌNG: CẬP NHẬT X_TRAIN_COLUMNS_ORDERED !!! -----
# Danh sách này PHẢI KHỚP CHÍNH XÁC với tên và thứ tự các cột
# mà mô hình của bạn đã được huấn luyện (SAU KHI One-Hot Encoding và Scaling).
# Dưới đây là một VÍ DỤ dựa trên các tùy chọn ở trên.
# BẠN CẦN THAY THẾ BẰNG CÁC TÊN CỘT ONE-HOT ENCODED THỰC TẾ TỪ DỮ LIỆU CỦA BẠN.
X_TRAIN_COLUMNS_ORDERED = NUMERICAL_FEATURES_RAW + \
                          [f'Project Type_{val}' for val in PROJECT_TYPES_OPTIONS] + \
                          [f'Language Type_{val}' for val in LANGUAGE_TYPES_OPTIONS] + \
                          [f'Primary Programming Language_{val}' for val in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS] + \
                          [f'Count Approach_{val}' for val in COUNT_APPROACH_OPTIONS] + \
                          [f'Application Group_{val}' for val in APPLICATION_GROUP_OPTIONS] + \
                          [f'Application Type_{val}' for val in APPLICATION_TYPES_OPTIONS] + \
                          [f'Development Type_{val}' for val in DEVELOPMENT_TYPES_OPTIONS]


# Loại bỏ các cột có thể không được tạo ra nếu giá trị không có trong dữ liệu huấn luyện
# Ví dụ: nếu 'Project Type_Khác' không có trong dữ liệu huấn luyện, nó sẽ không được tạo ra bởi encoder.
# Cách tốt nhất là lấy danh sách cột này từ `encoder.get_feature_names_out()` hoặc từ `X_train.columns` sau khi xử lý.


# --- Hàm Tính Toán (đã có từ trước) ---
def calculate_metrics(size_metric_choice, size_metric_value, language):
    calculated_loc = 0.0
    calculated_fp = 0.0
    calculated_ucp = 0.0
    estimated_effort_pm = 0.0  # Person-Months
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
                calculated_ucp = (effort_pm_from_loc * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
    elif size_metric_choice == 'FP':
        calculated_fp = size_metric_value
        calculated_loc = calculated_fp * loc_fp_ratio
        kloc = calculated_loc / 1000
        if kloc > 0:
            effort_pm_from_fp_via_loc = COCOMO_A * (kloc ** COCOMO_B)
            if EFFORT_PER_UCP > 0 and HOURS_PER_PERSON_MONTH > 0:
                calculated_ucp = (effort_pm_from_fp_via_loc * HOURS_PER_PERSON_MONTH) / EFFORT_PER_UCP
    elif size_metric_choice == 'UCP':
        calculated_ucp = size_metric_value
        if EFFORT_PER_UCP > 0 and HOURS_PER_PERSON_MONTH > 0:
            effort_pm_from_ucp = (calculated_ucp * EFFORT_PER_UCP) / HOURS_PER_PERSON_MONTH
            if COCOMO_A > 0 and COCOMO_B != 0 and effort_pm_from_ucp > 0:
                base_cocomo = effort_pm_from_ucp / COCOMO_A
                if base_cocomo > 0:
                    kloc = base_cocomo ** (1 / COCOMO_B)
                    calculated_loc = kloc * 1000
                    if loc_fp_ratio > 0: calculated_fp = calculated_loc / loc_fp_ratio

    final_kloc = calculated_loc / 1000
    if final_kloc > 0:
        estimated_effort_pm = COCOMO_A * (final_kloc ** COCOMO_B)  # Effort này là Person-Months
        if estimated_effort_pm > 0:
            base_dev_time = estimated_effort_pm
            if base_dev_time > 0:
                estimated_dev_time_months = COCOMO_C * (base_dev_time ** COCOMO_D)
            if estimated_dev_time_months > 0:
                estimated_team_size = estimated_effort_pm / estimated_dev_time_months
            else:
                estimated_team_size = 1 if estimated_effort_pm > 0 else 0

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

    cocomo_params = {
        "Organic": (2.4, 1.05),
        "Semi-detached": (3.0, 1.12),
        "Embedded": (3.6, 1.20)
    }
    a, b = cocomo_params.get(project_type_cocomo, cocomo_params["Organic"])

    if kloc <= 0: return 0.0
    effort_pm = a * (kloc ** b) * effort_multipliers  # Effort này là Person-Months
    return round(effort_pm, 2)


# --- Hàm tải mô hình và preprocessors ---
def load_model_and_preprocessors():
    models = {}
    scaler = None
    # Giả sử bạn lưu từng encoder riêng lẻ hoặc một ColumnTransformer
    encoders = {}  # Ví dụ: {'Project Type': loaded_encoder_for_project_type, ...}
    # HOẶC column_transformer = joblib.load('column_transformer.pkl')

    model_files = {
        'Linear Regression': 'linear_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgb_regressor_model.pkl',
        'MLP Regressor': 'mlp_regressor_model.pkl'
    }

    try:
        # scaler = joblib.load('scaler.pkl') # Bỏ comment và sửa đường dẫn file
        st.sidebar.warning("Scaler (scaler.pkl) chưa được tải. Sử dụng dữ liệu chưa scale.")

        # Ví dụ tải từng encoder (CẦN ĐIỀU CHỈNH THEO CÁCH BẠN LƯU ENCODERS):
        # for col in CATEGORICAL_FEATURES_RAW:
        #     try:
        #         # Tên file encoder có thể khác nhau tùy theo cách bạn lưu
        #         encoders[col] = joblib.load(f'{col.lower().replace(" ", "_").replace("/", "_")}_encoder.pkl')
        #     except FileNotFoundError:
        #         st.sidebar.warning(f"Encoder cho cột '{col}' không tìm thấy. Mã hóa one-hot có thể không chính xác.")
        #         encoders[col] = None # Quan trọng: đánh dấu là None nếu không tải được
        st.sidebar.warning(
            "Các encoders chưa được tải. Mã hóa one-hot sẽ không chính xác. Cần cung cấp các file .pkl cho encoders.")

        for model_name, file_name in model_files.items():
            # models[model_name] = joblib.load(file_name) # Bỏ comment và sửa đường dẫn file
            st.sidebar.warning(f"Mô hình {model_name} ({file_name}) chưa được tải. Sẽ không có dự đoán từ mô hình này.")
            models[model_name] = None  # Placeholder nếu không tải được

    except FileNotFoundError as e:
        st.sidebar.error(f"Lỗi: Không tìm thấy file mô hình hoặc preprocessor. Vui lòng kiểm tra đường dẫn. {e}")
    except Exception as e:
        st.sidebar.error(f"Lỗi khi tải mô hình hoặc preprocessor: {e}")

    return models, scaler, encoders


# --- Giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("⚙️ Công cụ Ước tính Nỗ lực Phát triển Phần mềm v3 (Cột Thực Tế)")

# Khởi tạo session state
if 'ml_predictions_ph' not in st.session_state: st.session_state.ml_predictions_ph = None  # Lưu dự đoán ML (Person-Hours)
if 'cocomo_estimate_ph' not in st.session_state: st.session_state.cocomo_estimate_ph = None
if 'processed_input_df_display' not in st.session_state: st.session_state.processed_input_df_display = None
if 'raw_input_df_display' not in st.session_state: st.session_state.raw_input_df_display = None

# --- Sidebar ---
with st.sidebar:
    st.header("📊 Nhập Thông tin & Ước tính")
    st.markdown("---")

    # Các trường nhập liệu cơ bản cho tính toán sơ bộ
    size_metric_choice = st.selectbox(
        "Chọn chỉ số kích thước đầu vào chính:", ('LOC', 'FP', 'UCP'), key='size_metric_choice_v3'
    )

    size_metric_label = f"Nhập giá trị cho {size_metric_choice}:"
    if size_metric_choice == 'LOC':
        default_val_size, step_val_size = 10000.0, 1000.0
    elif size_metric_choice == 'FP':
        default_val_size, step_val_size = 200.0, 10.0
    else:
        default_val_size, step_val_size = 100.0, 5.0

    size_metric_value = st.number_input(
        size_metric_label, min_value=0.0, value=default_val_size, step=step_val_size, key='size_metric_value_v3',
        format="%.2f"
    )

    # Ngôn ngữ lập trình chính (vẫn cần cho tính toán LOC/FP)
    selected_primary_programming_language = st.selectbox(
        "Ngôn ngữ lập trình chính (cho quy đổi LOC/FP):", PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS,
        index=PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS.index(
            'Java') if 'Java' in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS else 0,
        key='selected_primary_programming_language_v3'
    )

    # Tính toán các chỉ số cơ bản
    (calculated_loc, calculated_fp, calculated_ucp,
     estimated_effort_pm,  # Đây là Person-Months từ COCOMO cơ bản
     estimated_dev_time_months,
     estimated_team_size) = calculate_metrics(
        size_metric_choice, size_metric_value, selected_primary_programming_language
    )

    st.markdown("---")
    st.subheader("📈 Các Chỉ số Kích thước Ước tính:")
    col_loc_ucp, col_fp_empty = st.columns(2)
    with col_loc_ucp:
        st.metric(label="LOC", value=f"{calculated_loc:,.0f}",
                  delta="Tính toán" if size_metric_choice != 'LOC' else "Đầu vào", delta_color="off")
        st.metric(label="UCP", value=f"{calculated_ucp:,.0f}",
                  delta="Tính toán" if size_metric_choice != 'UCP' else "Đầu vào", delta_color="off")
    with col_fp_empty:
        st.metric(label="FP", value=f"{calculated_fp:,.0f}",
                  delta="Tính toán" if size_metric_choice != 'FP' else "Đầu vào", delta_color="off")

    st.markdown("---")
    st.subheader("⏱️ Ước tính Sơ bộ (COCOMO cơ bản):")
    col_e_pm, col_t_m, col_s_p = st.columns(3)
    with col_e_pm:
        st.metric(label="Nỗ lực (Person-Months)", value=f"{estimated_effort_pm:,.1f}")  # Hiển thị PM
    with col_t_m:
        st.metric(label="T.Gian P.Triển (Tháng)", value=f"{estimated_dev_time_months:,.1f}")
    with col_s_p:
        st.metric(label="Quy mô Nhóm (Người)", value=f"{estimated_team_size:,.1f}")

    st.markdown("---")
    st.subheader("📋 Thông tin Chi tiết Dự án (cho Model ML):")
    # Các trường nhập liệu mới dựa trên cột thực tế
    # Lưu ý: các giá trị LOC, FP, UCP, Dev Time, Team Size sẽ lấy từ calculated_* và estimated_* ở trên

    input_project_type = st.selectbox("Project Type:", PROJECT_TYPES_OPTIONS, key='input_project_type_v3')
    input_language_type = st.selectbox("Language Type:", LANGUAGE_TYPES_OPTIONS, key='input_language_type_v3')
    # Primary Programming Language cho model ML có thể khác với cái dùng để quy đổi LOC/FP nếu cần
    input_primary_lang_model = st.selectbox("Primary Programming Language (cho Model ML):",
                                            PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS,
                                            index=PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS.index(
                                                selected_primary_programming_language) if selected_primary_programming_language in PRIMARY_PROGRAMMING_LANGUAGES_OPTIONS else 0,
                                            key='input_primary_lang_model_v3')
    input_count_approach = st.selectbox("Count Approach:", COUNT_APPROACH_OPTIONS, key='input_count_approach_v3')
    input_app_group = st.selectbox("Application Group:", APPLICATION_GROUP_OPTIONS, key='input_app_group_v3')
    input_app_type = st.selectbox("Application Type:", APPLICATION_TYPES_OPTIONS, key='input_app_type_v3')
    input_dev_type = st.selectbox("Development Type:", DEVELOPMENT_TYPES_OPTIONS, key='input_dev_type_v3')

    st.markdown("---")
    if st.button("🚀 Dự đoán Effort Chính (ML & COCOMO II)", key='predict_effort_button_v3'):
        # 1. Thu thập tất cả các giá trị đầu vào cho DataFrame
        input_data_for_df = {
            'LOC': calculated_loc,
            'FP': calculated_fp,
            'UCP': calculated_ucp,
            'Development Time (months)': estimated_dev_time_months,
            'Team Size': estimated_team_size,
            'Project Type': input_project_type,
            'Language Type': input_language_type,
            'Primary Programming Language': input_primary_lang_model,  # Sử dụng ngôn ngữ đã chọn cho model
            'Count Approach': input_count_approach,
            'Application Group': input_app_group,
            'Application Type': input_app_type,
            'Development Type': input_dev_type
        }
        input_df_raw = pd.DataFrame([input_data_for_df])
        st.session_state.raw_input_df_display = input_df_raw.copy()

        # 2. Tải mô hình và preprocessors
        models, scaler, encoders = load_model_and_preprocessors()

        # 3. Tiền xử lý input DataFrame
        input_df_processed = input_df_raw.copy()
        processed_successfully = True

        # One-Hot Encoding
        if encoders:  # Chỉ thực hiện nếu dict encoders có gì đó (dù là None)
            for col in CATEGORICAL_FEATURES_RAW:  # Lặp qua các cột cần mã hóa
                if col in input_df_processed.columns:  # Kiểm tra cột có tồn tại trong df không
                    encoder_for_col = encoders.get(col)  # Lấy encoder tương ứng
                    if encoder_for_col is not None:
                        try:
                            encoded_data = encoder_for_col.transform(input_df_processed[[col]])
                            encoded_cols = encoder_for_col.get_feature_names_out([col])
                            encoded_df = pd.DataFrame(encoded_data, index=input_df_processed.index,
                                                      columns=encoded_cols)
                            input_df_processed = pd.concat([input_df_processed.drop(col, axis=1), encoded_df], axis=1)
                        except Exception as e:
                            st.error(f"Lỗi khi áp dụng OneHotEncoder đã tải cho cột '{col}': {e}.")
                            processed_successfully = False;
                            break
                    # else: st.warning(f"Không có encoder đã tải cho cột '{col}'. Bỏ qua mã hóa cột này bằng encoder đã lưu.")
                # else: st.warning(f"Cột '{col}' không có trong input_df_processed để mã hóa.")

        # Nếu không có encoder đã tải hoặc có lỗi, thử pd.get_dummies như một fallback
        # Điều này cần X_TRAIN_COLUMNS_ORDERED phải được định nghĩa rất cẩn thận
        # để khớp với kết quả của pd.get_dummies trên dữ liệu huấn luyện.
        if not encoders or not processed_successfully:  # Nếu không có dict encoders hoặc đã lỗi
            st.warning(
                "Không có encoders được tải đầy đủ hoặc có lỗi. Sử dụng pd.get_dummies() cho tất cả các cột phân loại. Đảm bảo X_TRAIN_COLUMNS_ORDERED khớp với cách này.")
            try:
                input_df_processed = pd.get_dummies(input_df_processed, columns=CATEGORICAL_FEATURES_RAW,
                                                    dummy_na=False)
                processed_successfully = True  # Giả sử thành công nếu không có lỗi
            except Exception as e:
                st.error(f"Lỗi khi sử dụng pd.get_dummies: {e}");
                processed_successfully = False

        # Áp dụng Chuẩn hóa (StandardScaler)
        if scaler and processed_successfully:
            try:
                # Chỉ scale các cột số có trong input_df_processed và NUMERICAL_FEATURES_RAW
                cols_to_scale = [col for col in NUMERICAL_FEATURES_RAW if col in input_df_processed.columns]
                if cols_to_scale:
                    input_df_processed[cols_to_scale] = scaler.transform(input_df_processed[cols_to_scale])
                # else: st.warning("Không có cột số nào để scale trong dữ liệu đầu vào sau OHE.")
            except ValueError as ve:
                st.error(
                    f"Lỗi ValueError khi áp dụng StandardScaler: {ve}. Có thể do số lượng features không khớp. Kiểm tra các cột: {cols_to_scale}")
                processed_successfully = False
            except Exception as e:
                st.error(f"Lỗi khi áp dụng StandardScaler: {e}.")
                processed_successfully = False
        # else:
        #     if not scaler: st.warning("Scaler chưa được tải. Dữ liệu số sẽ không được chuẩn hóa.")

        # Đảm bảo thứ tự các cột và sự tồn tại của tất cả các cột từ X_TRAIN_COLUMNS_ORDERED
        input_df_final_for_model = pd.DataFrame()  # Khởi tạo df rỗng
        if processed_successfully:
            # Thêm các cột bị thiếu (nếu có sau OHE) với giá trị 0
            for col_model_expected in X_TRAIN_COLUMNS_ORDERED:
                if col_model_expected not in input_df_processed.columns:
                    input_df_processed[col_model_expected] = 0
            try:
                # Chọn và sắp xếp lại các cột theo đúng thứ tự mô hình mong đợi
                input_df_final_for_model = input_df_processed[X_TRAIN_COLUMNS_ORDERED]
                st.session_state.processed_input_df_display = input_df_final_for_model.copy()
            except KeyError as e:
                st.error(
                    f"Lỗi KeyError khi chọn các cột cuối cùng cho mô hình: {e}. Điều này thường xảy ra nếu X_TRAIN_COLUMNS_ORDERED chứa tên cột không có trong input_df_processed sau khi OHE. Hãy kiểm tra lại X_TRAIN_COLUMNS_ORDERED và quá trình OHE.")
                processed_successfully = False
            except Exception as e:
                st.error(f"Lỗi không xác định khi sắp xếp lại các cột cuối cùng: {e}")
                processed_successfully = False

        # 4. Thực hiện dự đoán (Effort là Person-Months từ mô hình)
        ml_predictions_pm = {}
        if processed_successfully and not input_df_final_for_model.empty:
            for model_name, model_object in models.items():
                if model_object is not None:
                    try:
                        prediction_pm = model_object.predict(input_df_final_for_model)
                        ml_predictions_pm[model_name] = round(prediction_pm[0],
                                                              2)  # Giả sử mô hình trả về Person-Months
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán với mô hình {model_name}: {e}")
                        ml_predictions_pm[model_name] = "Lỗi dự đoán"
                else:
                    ml_predictions_pm[model_name] = "Mô hình chưa tải"
        else:
            st.warning("Không thể thực hiện dự đoán ML do lỗi trong quá trình chuẩn bị dữ liệu.")
            for model_name_key in models.keys(): ml_predictions_pm[model_name_key] = "Lỗi dữ liệu"

        # Chuyển đổi dự đoán ML từ Person-Months sang Person-Hours và lưu vào session_state
        st.session_state.ml_predictions_ph = {
            name: (effort_pm * HOURS_PER_PERSON_MONTH if isinstance(effort_pm, (int, float)) else effort_pm)
            for name, effort_pm in ml_predictions_pm.items()
        }

        # 5. Tính toán COCOMO II (Effort là Person-Months)
        kloc_for_cocomo = calculated_loc / 1000
        # Ánh xạ project_type từ UI sang loại COCOMO (ví dụ)
        cocomo_project_type_map = {
            'Phát triển mới': "Organic", 'Nâng cấp lớn': "Semi-detached",
            'Bảo trì': "Organic", 'Tái cấu trúc': "Semi-detached",
            'Tích hợp hệ thống': "Embedded", 'Khác': "Organic"
        }
        cocomo_type_for_calc = cocomo_project_type_map.get(input_project_type, "Organic")  # Sử dụng input_project_type
        cocomo_effort_pm_estimated = estimate_cocomo_effort(kloc_for_cocomo, project_type_cocomo=cocomo_type_for_calc)
        # Chuyển đổi COCOMO từ Person-Months sang Person-Hours và lưu vào session_state
        st.session_state.cocomo_estimate_ph = round(cocomo_effort_pm_estimated * HOURS_PER_PERSON_MONTH, 0)

        st.success("Đã thực hiện dự đoán Effort!")

# --- Khu vực chính ---
main_area = st.container()
with main_area:
    st.header("🔍 Kết quả Ước tính Chi tiết và Phân tích")

    if st.session_state.raw_input_df_display is not None:
        st.subheader("Dữ liệu đầu vào thô (trước khi xử lý cho ML):")
        st.dataframe(st.session_state.raw_input_df_display, use_container_width=True)

    if st.session_state.processed_input_df_display is not None:
        st.subheader("Dữ liệu đầu vào đã xử lý (cho mô hình ML):")
        st.dataframe(st.session_state.processed_input_df_display, use_container_width=True)
        st.caption(
            f"Số cột mong đợi bởi mô hình: {len(X_TRAIN_COLUMNS_ORDERED)}. Số cột thực tế truyền vào: {st.session_state.processed_input_df_display.shape[1]}")

    if st.session_state.ml_predictions_ph:  # Kiểm tra session state mới
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
                            st.metric(label=f"{model_name} (PH)", value=str(effort_value_ph))  # Hiển thị lỗi nếu có
        st.markdown("---")

        # Biểu đồ So sánh Tổng hợp
        st.subheader("📈 Biểu đồ So sánh Effort Tổng hợp (person-hours)")
        comparison_data_for_chart = {}
        for model_name_chart, effort_ph_chart in current_ml_predictions_ph.items():
            if isinstance(effort_ph_chart, (int, float)):
                comparison_data_for_chart[model_name_chart] = effort_ph_chart

        if st.session_state.cocomo_estimate_ph is not None and isinstance(st.session_state.cocomo_estimate_ph,
                                                                          (int, float)):
            comparison_data_for_chart["COCOMO II"] = st.session_state.cocomo_estimate_ph

        if comparison_data_for_chart:
            df_comparison_chart = pd.DataFrame(list(comparison_data_for_chart.items()),
                                               columns=['Phương pháp', 'Effort (Person-Hours)'])
            df_comparison_chart = df_comparison_chart.sort_values(by='Effort (Person-Hours)', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 7))  # Tăng chiều cao một chút
            bars = ax.bar(df_comparison_chart['Phương pháp'], df_comparison_chart['Effort (Person-Hours)'],
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])  # Màu sắc khác nhau

            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0,
                         yval + 0.02 * max(df_comparison_chart['Effort (Person-Hours)']), f'{yval:,.0f}', ha='center',
                         va='bottom', fontsize=9)

            ax.set_ylabel('Effort Ước tính (Person-Hours)', fontsize=12)
            ax.set_title('So sánh Effort Ước tính giữa các Phương pháp', fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)  # Thêm lưới ngang mờ
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Không có đủ dữ liệu hợp lệ để vẽ biểu đồ so sánh tổng hợp.")

    # Hiển thị COCOMO metric riêng lẻ
    if st.session_state.cocomo_estimate_ph is not None:
        st.subheader("⚙️ Ước tính Effort từ COCOMO II (person-hours) - Chi tiết")
        st.metric(label="COCOMO II Effort (PH)", value=f"{st.session_state.cocomo_estimate_ph:,.0f}")

    if not st.session_state.ml_predictions_ph and st.session_state.cocomo_estimate_ph is None:
        st.info("Nhập thông tin ở thanh bên trái và nhấn '🚀 Dự đoán Effort Chính (ML & COCOMO II)' để xem kết quả.")

    st.markdown("---")
    st.subheader("📝 Chỉ số Đánh giá Mô hình (Tùy chọn)")
    st.info(
        "Để hiển thị các chỉ số đánh giá (ví dụ: MAE, RMSE, R²) của các mô hình, "
        "bạn cần tải chúng từ quá trình huấn luyện mô hình và hiển thị tại đây."
    )

    st.markdown("---")
    st.subheader("Hướng dẫn Tiếp theo và Lưu ý Quan Trọng")
    st.markdown("""
    1.  **CUNG CẤP FILE MÔ HÌNH VÀ PREPROCESSORS:** Bỏ comment các dòng `joblib.load(...)` trong hàm `load_model_and_preprocessors()` và cung cấp đường dẫn chính xác đến các file `.pkl` của bạn (scaler, encoders, models).
    2.  **XÁC ĐỊNH `X_TRAIN_COLUMNS_ORDERED` CHÍNH XÁC:** Đây là bước **CỰC KỲ QUAN TRỌNG**. Danh sách này phải khớp hoàn toàn với tên và thứ tự các cột của dữ liệu bạn đã dùng để huấn luyện mô hình (sau khi đã one-hot encoding và scaling). Hãy kiểm tra và cập nhật cẩn thận dựa trên output của `encoder.get_feature_names_out()` hoặc `X_train.columns` của bạn.
    3.  **KIỂM TRA LOGIC ONE-HOT ENCODING (OHE):** Cách bạn lưu và tải encoders (từng cái một hay dùng `ColumnTransformer`) phải nhất quán. Logic OHE trong code cần phản ánh điều đó.
    4.  **ĐƠN VỊ EFFORT:** Đảm bảo rằng mô hình ML của bạn được huấn luyện để dự đoán effort theo đơn vị **Person-Months**. Code hiện tại giả định điều này và sau đó chuyển đổi sang Person-Hours để hiển thị. Nếu mô hình của bạn dự đoán trực tiếp Person-Hours, bạn cần điều chỉnh logic chuyển đổi.
    5.  **CÁC LỰA CHỌN CHO SELECTBOX:** Cập nhật các biến `..._OPTIONS` (ví dụ: `PROJECT_TYPES_OPTIONS`) với các giá trị thực tế từ dữ liệu của bạn để các dropdown menu hiển thị đúng.
    """)

# Để chạy ứng dụng này:
# 1. Cài đặt: streamlit, pandas, numpy, scikit-learn, joblib, matplotlib, (xgboost nếu dùng)
# 2. Lưu file: app_ml_real_cols.py (hoặc tên khác)
# 3. Chuẩn bị các file .pkl (mô hình, scaler, encoders) và cập nhật đường dẫn trong code.
# 4. CẬP NHẬT `X_TRAIN_COLUMNS_ORDERED` VÀ CÁC `..._OPTIONS` CHO CHÍNH XÁC.
# 5. Chạy lệnh: streamlit run app_ml_real_cols.py
