# Import các thư viện cần thiết
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Để tải mô hình và preprocessors
import matplotlib.pyplot as plt  # Thêm thư viện matplotlib
import os  # Thêm để làm việc với đường dẫn file
from collections import OrderedDict  # Thêm để giữ thứ tự models
import math  # Cần cho COCOMO
import traceback  # Thêm để in lỗi chi tiết

# Import các lớp cần thiết từ scikit-learn (nếu cần tham chiếu kiểu)
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder  # Hoặc StandardScaler nếu bạn dùng
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
except ImportError as e:
    st.error(f"Lỗi Import thư viện scikit-learn hoặc xgboost: {e}. Hãy đảm bảo chúng đã được cài đặt.")
    st.stop()

# --- Cấu hình Trang ---
st.set_page_config(page_title="So sánh Ước tính Effort Phần mềm", layout="wide")
st.title("Ứng dụng So sánh Ước tính Effort Phần mềm 📊")
st.write("""
Nhập thông tin dự án để nhận ước tính effort (person-hours) từ nhiều mô hình Machine Learning
và các phương pháp truyền thống (COCOMO II Basic, Function Points, Use Case Points).
""")

# --- Định nghĩa đường dẫn và Hằng số ---
# QUAN TRỌNG: Đảm bảo các file .joblib (preprocessor.joblib, feature_names.joblib, và các model .joblib)
# nằm trong thư mục được chỉ định bởi OUTPUT_DIR.
# Mặc định là thư mục hiện tại nơi chạy app.py.
OUTPUT_DIR = "."
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.joblib")  # Tên features SAU KHI preprocessor transform
MODEL_PATHS = OrderedDict([
    ('Linear Regression', os.path.join(OUTPUT_DIR, "linear_regression_model.joblib")),
    ('Decision Tree', os.path.join(OUTPUT_DIR, "decision_tree_model.joblib")),
    ('Random Forest', os.path.join(OUTPUT_DIR, "random_forest_model.joblib")),
    ('XGBoost', os.path.join(OUTPUT_DIR, "xgboost_model.joblib")),
    ('MLP Regressor', os.path.join(OUTPUT_DIR, "mlp_regressor_model.joblib"))
])

HOURS_PER_PERSON_MONTH = 152  # Số giờ làm việc trung bình mỗi tháng cho một người


# --- Hàm tải Artifacts (Dựa trên app.py bạn cung cấp) ---
@st.cache_resource  # Cache để không tải lại mỗi lần rerun
def load_all_artifacts_from_files(preprocessor_path, features_path, model_paths_dict):
    """
    Tải preprocessor, feature names (sau xử lý), các mô hình ML,
    và trích xuất thứ tự cột gốc cùng các tùy chọn cho cột phân loại từ preprocessor.
    Trả về: preprocessor, feature_names_processed, loaded_models, original_cols_order, categorical_features_options, load_status_flag
    """
    loaded_models = OrderedDict()
    preprocessor = None
    feature_names_processed = None  # Tên features SAU KHI preprocessor transform
    categorical_features_options = {}  # Lưu các categories cho mỗi cột categorical gốc
    original_cols_order = []  # Thứ tự các cột gốc mà preprocessor mong đợi
    all_loaded_successfully = True  # Cờ theo dõi trạng thái tải

    # 1. Tải Preprocessor
    if not os.path.exists(preprocessor_path):
        st.error(f"LỖI: Không tìm thấy file preprocessor tại '{preprocessor_path}'.")
        return None, None, None, None, None, False  # Thêm cờ trạng thái tải
    try:
        preprocessor = joblib.load(preprocessor_path)
        st.sidebar.success(f"Preprocessor: Tải thành công.")

        # Trích xuất thông tin từ Preprocessor
        try:
            # Tìm transformer cho numerical và categorical features
            # Tên 'num' và 'cat' phải khớp với tên bạn đặt trong ColumnTransformer
            num_transformer_info = next(t for t in preprocessor.transformers_ if t[0] == 'num')
            cat_transformer_info = next(t for t in preprocessor.transformers_ if t[0] == 'cat')

            original_num_features = list(num_transformer_info[2])  # Danh sách tên cột số gốc
            original_cat_features = list(cat_transformer_info[2])  # Danh sách tên cột phân loại gốc
            original_cols_order = original_num_features + original_cat_features

            # Trích xuất categories từ OneHotEncoder bên trong pipeline của 'cat'
            cat_pipeline = preprocessor.named_transformers_['cat']  # Hoặc tên pipeline của bạn
            onehot_encoder = cat_pipeline.named_steps['onehot']  # Hoặc tên bước onehot của bạn

            if hasattr(onehot_encoder, 'categories_'):
                if len(onehot_encoder.categories_) == len(original_cat_features):
                    for i, feature_name in enumerate(original_cat_features):
                        categories = onehot_encoder.categories_[i]
                        # Loại bỏ np.nan nếu có trong categories (thường do SimpleImputer tạo ra)
                        # và chuyển thành string để đảm bảo tính nhất quán cho selectbox
                        cleaned_categories = [str(cat) for cat in categories if
                                              not (isinstance(cat, float) and np.isnan(cat))]
                        categorical_features_options[feature_name] = cleaned_categories
                    st.sidebar.caption("Đã trích xuất tùy chọn cho cột phân loại.")
                else:
                    st.error(
                        f"Lỗi trích xuất preprocessor: Số lượng categories ({len(onehot_encoder.categories_)}) không khớp số cột phân loại ({len(original_cat_features)}).")
                    all_loaded_successfully = False
            else:
                st.error("Lỗi trích xuất preprocessor: Không tìm thấy 'categories_' trong OneHotEncoder.")
                all_loaded_successfully = False
        except StopIteration:
            st.error(
                "Lỗi trích xuất preprocessor: Không tìm thấy transformer 'num' hoặc 'cat'. Kiểm tra tên trong ColumnTransformer.")
            all_loaded_successfully = False
        except KeyError as ke:
            st.error(f"Lỗi trích xuất preprocessor: Không tìm thấy step '{ke}' trong pipeline. Kiểm tra tên step.")
            all_loaded_successfully = False
        except Exception as e_extract:
            st.error(f"Lỗi khi trích xuất thông tin từ preprocessor: {e_extract}")
            all_loaded_successfully = False
            print(traceback.format_exc())
    except Exception as e_load_prep:
        st.error(f"Lỗi nghiêm trọng khi tải preprocessor: {e_load_prep}")
        print(traceback.format_exc())
        return None, None, None, None, None, False

    # 2. Tải Feature Names (sau khi xử lý bởi preprocessor)
    if not os.path.exists(features_path):
        st.error(f"LỖI: Không tìm thấy file tên đặc trưng (đã xử lý) tại '{features_path}'.")
        all_loaded_successfully = False
    else:
        try:
            feature_names_processed = joblib.load(features_path)
            if isinstance(feature_names_processed, np.ndarray):
                feature_names_processed = feature_names_processed.tolist()
            if not isinstance(feature_names_processed, list):
                st.warning(
                    f"Định dạng feature_names_processed không phải list (kiểu: {type(feature_names_processed)}). Cố gắng chuyển đổi.")
                try:
                    feature_names_processed = list(feature_names_processed)
                except TypeError:
                    st.error("Không thể chuyển đổi feature_names_processed thành list.")
                    all_loaded_successfully = False
            if feature_names_processed and all_loaded_successfully:  # Chỉ thông báo nếu chưa có lỗi
                st.sidebar.success(f"Tên đặc trưng (đã xử lý): Tải thành công.")
        except Exception as e_load_feat:
            st.error(f"Lỗi khi tải feature_names_processed: {e_load_feat}")
            all_loaded_successfully = False

    # 3. Tải các Mô hình ML
    models_actually_loaded_count = 0
    for name, path in model_paths_dict.items():
        if not os.path.exists(path):
            st.warning(f"Cảnh báo: Không tìm thấy file mô hình '{name}' tại '{path}'. Bỏ qua.")
            continue
        try:
            model = joblib.load(path)
            loaded_models[name] = model
            models_actually_loaded_count += 1
        except Exception as e_load_model:
            st.warning(f"Lỗi khi tải mô hình {name} từ '{path}': {e_load_model}. Bỏ qua.")

    if models_actually_loaded_count > 0:
        st.sidebar.success(
            f"Mô hình ML: Tải thành công {models_actually_loaded_count}/{len(model_paths_dict)} mô hình.")
    else:
        st.error("LỖI: Không tải được bất kỳ mô hình Machine Learning nào.")
        # Không nhất thiết phải đặt all_loaded_successfully = False ở đây nếu vẫn muốn dùng mô hình truyền thống

    # Kiểm tra cuối cùng trước khi trả về
    if not preprocessor or not feature_names_processed or not original_cols_order:
        st.error("Thiếu một hoặc nhiều thành phần ML quan trọng (preprocessor, feature_names, original_cols_order).")
        all_loaded_successfully = False
        # Không trả về categorical_features_options nếu original_cols_order rỗng hoặc preprocessor lỗi
        if not original_cols_order or not preprocessor: categorical_features_options = {}

    return preprocessor, feature_names_processed, loaded_models, original_cols_order, categorical_features_options, all_loaded_successfully


# --- Thực hiện tải artifacts ---
preprocessor, feature_names_processed_from_file, ml_models_loaded, \
    original_cols_input_order, categorical_options_from_preprocessor, artifacts_load_status = load_all_artifacts_from_files(
    PREPROCESSOR_PATH, FEATURES_PATH, MODEL_PATHS
)


# --- Hàm tính toán cho mô hình truyền thống (Từ app.py, trả về Person-Hours) ---
def calculate_cocomo_basic(loc, mode, eaf, hrs_per_month):
    if loc <= 0: return "Lỗi (LOC <= 0)"
    if hrs_per_month <= 0: return "Lỗi (Giờ/Tháng <= 0)"
    kloc = loc / 1000.0
    params = {"Organic": {"a": 2.4, "b": 1.05}, "Semi-detached": {"a": 3.0, "b": 1.12},
              "Embedded": {"a": 3.6, "b": 1.20}}
    if mode not in params: return "Lỗi (Chế độ không hợp lệ)"
    a, b = params[mode]["a"], params[mode]["b"]
    try:
        person_months = a * (kloc ** b) * eaf
        return max(0.0, round(person_months * hrs_per_month, 2))
    except Exception as e:
        return f"Lỗi tính toán COCOMO: {e}"


def calculate_fp_effort(fp, hrs_per_fp):
    if fp <= 0: return "Lỗi (FP <= 0)"
    if hrs_per_fp <= 0: return "Lỗi (Giờ/FP <= 0)"
    try:
        return max(0.0, round(fp * hrs_per_fp, 2))
    except Exception as e:
        return f"Lỗi tính toán FP: {e}"


def calculate_ucp_effort(ucp, hrs_per_ucp):
    if ucp <= 0: return "Lỗi (UCP <= 0)"
    if hrs_per_ucp <= 0: return "Lỗi (Giờ/UCP <= 0)"
    try:
        return max(0.0, round(ucp * hrs_per_ucp, 2))
    except Exception as e:
        return f"Lỗi tính toán UCP: {e}"


# --- Giao diện Nhập liệu Sidebar ---
st.sidebar.header("Nhập Thông tin Dự án")
input_values_for_ml = {}  # Dictionary cho các giá trị sẽ được truyền vào preprocessor

# Các trường nhập liệu cơ bản (LOC, FP, UCP) - Dùng cho cả ML (nếu có trong original_cols_input_order) và truyền thống
st.sidebar.subheader("Kích thước Dự án (Ước tính)")
col_loc_fp, col_ucp_empty = st.sidebar.columns(2)
with col_loc_fp:
    # LOC (Luôn hiển thị vì cần cho COCOMO)
    loc_input_val = st.number_input("Lines of Code (LOC)", min_value=0, value=10000, step=100, key="loc_input_v_adv")
    if original_cols_input_order and 'LOC' in original_cols_input_order:
        input_values_for_ml['LOC'] = loc_input_val

    # FP (Luôn hiển thị vì cần cho tính toán FP)
    fp_input_val = st.number_input("Function Points (FP)", min_value=0, value=100, step=10, key="fp_input_v_adv")
    if original_cols_input_order and 'FP' in original_cols_input_order:
        input_values_for_ml['FP'] = fp_input_val
with col_ucp_empty:
    # UCP (Luôn hiển thị vì cần cho tính toán UCP)
    ucp_input_val = st.number_input("Use Case Points (UCP)", min_value=0.0, value=100.0, step=10.0, format="%.2f",
                                    key="ucp_input_v_adv")
    if original_cols_input_order and 'UCP' in original_cols_input_order:
        input_values_for_ml['UCP'] = ucp_input_val

# Các trường nhập liệu cho ML (tạo động dựa trên original_cols_input_order và categorical_options_from_preprocessor)
if artifacts_load_status and preprocessor and original_cols_input_order:
    st.sidebar.subheader("Đặc trưng Dự án (Cho Model ML)")

    # Lấy danh sách cột số và phân loại gốc từ original_cols_input_order và categorical_options_from_preprocessor
    original_numerical_cols_for_ui = [col for col in original_cols_input_order if
                                      col not in categorical_options_from_preprocessor and col not in ['LOC', 'FP',
                                                                                                       'UCP']]
    original_categorical_cols_for_ui = [col for col in original_cols_input_order if
                                        col in categorical_options_from_preprocessor]

    # Input cho các cột số (trừ LOC, FP, UCP đã có ở trên)
    if original_numerical_cols_for_ui:
        st.sidebar.markdown("**Đặc trưng dạng số:**")
        num_cols_display = min(2, len(original_numerical_cols_for_ui)) if original_numerical_cols_for_ui else 1
        cols_num_ui = st.sidebar.columns(num_cols_display)

        for i, col_name in enumerate(original_numerical_cols_for_ui):
            current_col_container = cols_num_ui[i % num_cols_display]
            with current_col_container:
                default_val = 10;
                step_val = 1;
                min_val = 0
                if "month" in col_name.lower() or "time" in col_name.lower():
                    default_val = 6; min_val = 1
                elif "size" in col_name.lower():
                    default_val = 5; min_val = 1
                input_values_for_ml[col_name] = st.number_input(f"{col_name}", min_value=min_val, value=default_val,
                                                                step=step_val, key=f"ml_num_{col_name}")

    # Input cho các cột phân loại
    if original_categorical_cols_for_ui:
        st.sidebar.markdown("**Đặc trưng dạng phân loại:**")
        num_cat_cols_display = min(2, len(original_categorical_cols_for_ui)) if original_categorical_cols_for_ui else 1
        cols_cat_ui = st.sidebar.columns(num_cat_cols_display)

        for i, col_name in enumerate(original_categorical_cols_for_ui):
            current_col_cat_container = cols_cat_ui[i % num_cat_cols_display]
            with current_col_cat_container:
                options = categorical_options_from_preprocessor.get(col_name, [])
                if options:
                    default_index = 0
                    input_values_for_ml[col_name] = st.selectbox(f"{col_name}", options=options, index=default_index,
                                                                 key=f"ml_cat_{col_name}")
                else:  # Fallback nếu không có options (dù không nên xảy ra nếu preprocessor trích xuất đúng)
                    st.sidebar.warning(
                        f"Không có tùy chọn cho '{col_name}'. Nhập thủ công (nếu preprocessor của bạn có thể xử lý giá trị mới hoặc bạn có imputer).")
                    input_values_for_ml[col_name] = st.text_input(f"{col_name} (nhập tay)",
                                                                  key=f"ml_cat_text_{col_name}")
else:
    st.sidebar.warning(
        "Không thể tải đầy đủ preprocessor hoặc thông tin cột. Phần nhập liệu chi tiết cho ML bị hạn chế/vô hiệu hóa.")
    st.sidebar.info("Vui lòng kiểm tra các file: preprocessor.joblib, feature_names.joblib trong thư mục OUTPUT_DIR.")

# --- Widget nhập liệu cho Mô hình Truyền thống ---
st.sidebar.subheader("Tham số cho Mô hình Truyền thống")
# COCOMO II Basic
st.sidebar.markdown("**COCOMO II (Basic)**")
cocomo_mode_input = st.sidebar.selectbox("Chế độ Dự án COCOMO", ["Organic", "Semi-detached", "Embedded"],
                                         key="cocomo_mode_input")
eaf_input = st.sidebar.number_input("Hệ số Điều chỉnh Nỗ lực (EAF)", min_value=0.1, value=1.0, step=0.1, format="%.2f",
                                    key="eaf_input", help="Effort Adjustment Factor. 1.0 là nominal.")

# Function Points
st.sidebar.markdown("**Function Points (FP)**")
hours_per_fp_input = st.sidebar.number_input("Năng suất (giờ/FP)", min_value=0.1, value=10.0, step=0.5, format="%.1f",
                                             key="hrs_fp_input")

# Use Case Points
st.sidebar.markdown("**Use Case Points (UCP)**")
hours_per_ucp_input = st.sidebar.number_input("Năng suất (giờ/UCP)", min_value=0.1, value=20.0, step=1.0, format="%.1f",
                                              key="hrs_ucp_input")

# --- Nút Dự đoán/Tính toán ---
calculate_button = st.sidebar.button("📊 Ước tính & So sánh Effort", use_container_width=True, type="primary")

# --- Khởi tạo session state cho kết quả (nếu chưa có) ---
if 'raw_input_df_display' not in st.session_state:
    st.session_state.raw_input_df_display = None
if 'processed_input_df_display' not in st.session_state:
    st.session_state.processed_input_df_display = None

# --- Xử lý và Hiển thị Kết quả ---
if calculate_button:
    st.divider()
    st.subheader("📊 Kết quả Ước tính Effort Tổng hợp (Person-Hours)")

    all_estimation_results = OrderedDict()
    ml_error_messages = {}

    # --- 1. Dự đoán từ Mô hình Machine Learning ---
    if artifacts_load_status and preprocessor and feature_names_processed_from_file and ml_models_loaded and original_cols_input_order:
        st.markdown("#### 1. Dự đoán từ Mô hình Machine Learning")
        try:
            current_input_for_ml_df_values = {}
            missing_ml_inputs_runtime = []
            for col_orig_rt in original_cols_input_order:
                val = input_values_for_ml.get(col_orig_rt)
                current_input_for_ml_df_values[col_orig_rt] = val
                if val is None or (isinstance(val, str) and not val.strip()):  # Kiểm tra None hoặc chuỗi rỗng
                    is_categorical_rt = col_orig_rt in categorical_options_from_preprocessor
                    if not is_categorical_rt:  # Cột số
                        current_input_for_ml_df_values[col_orig_rt] = np.nan
                        # Với cột phân loại, preprocessor (với SimpleImputer(strategy='constant', fill_value='missing')
                    # và OneHotEncoder(handle_unknown='ignore')) sẽ xử lý.
                    # Nếu không có imputer cho categorical, giá trị None/rỗng có thể gây lỗi ở OHE trừ khi nó xử lý được.
                    missing_ml_inputs_runtime.append(f"{col_orig_rt} ({'phân loại' if is_categorical_rt else 'số'})")

            if missing_ml_inputs_runtime:
                st.caption(
                    f"ML Input: Giá trị None/rỗng/thiếu cho: {', '.join(missing_ml_inputs_runtime)}. Preprocessor sẽ cố gắng xử lý.")

            input_df_for_preprocessor = pd.DataFrame([current_input_for_ml_df_values],
                                                     columns=original_cols_input_order)
            st.session_state.raw_input_df_display = input_df_for_preprocessor.copy()

            input_processed_np_array = preprocessor.transform(input_df_for_preprocessor)

            if isinstance(feature_names_processed_from_file, list) and len(feature_names_processed_from_file) == \
                    input_processed_np_array.shape[1]:
                input_processed_final_df = pd.DataFrame(input_processed_np_array,
                                                        columns=feature_names_processed_from_file)
                st.session_state.processed_input_df_display = input_processed_final_df.copy()

                for model_name, loaded_model_object in ml_models_loaded.items():
                    try:
                        prediction_raw = loaded_model_object.predict(input_processed_final_df)
                        # Giả sử mô hình ML dự đoán Effort theo Person-Hours trực tiếp
                        prediction_value_ph = float(prediction_raw[0]) if prediction_raw.size > 0 else 0.0
                        all_estimation_results[f"ML: {model_name}"] = max(0.0, round(prediction_value_ph, 2))
                    except Exception as model_pred_e:
                        error_msg_detail = f"Lỗi dự đoán ({model_name}): {str(model_pred_e)}"
                        st.error(error_msg_detail)
                        all_estimation_results[f"ML: {model_name}"] = "Lỗi"
                        ml_error_messages[model_name] = str(model_pred_e)
            else:
                st.error(
                    f"Lỗi ML: Số tên đặc trưng đã xử lý ({len(feature_names_processed_from_file or [])}) không khớp số cột sau transform ({input_processed_np_array.shape[1]}).")
                for model_name_key_err in (ml_models_loaded.keys() if ml_models_loaded else MODEL_PATHS.keys()):
                    all_estimation_results[f"ML: {model_name_key_err}"] = "Lỗi (Cấu hình Feature)"
        except Exception as e_ml_main_process:
            st.error(f"Lỗi nghiêm trọng trong tiền xử lý/dự đoán ML: {e_ml_main_process}")
            for model_name_key_err_main in (ml_models_loaded.keys() if ml_models_loaded else MODEL_PATHS.keys()):
                all_estimation_results[f"ML: {model_name_key_err_main}"] = "Lỗi (Tiền xử lý)"
            print(traceback.format_exc())
    else:
        st.info("Dự đoán ML không thực hiện do thiếu thành phần hoặc lỗi tải artifacts.")

    # --- 2. Tính toán từ Mô hình Truyền thống (Đã trả về Person-Hours) ---
    st.markdown("#### 2. Tính toán từ Mô hình Truyền thống")
    traditional_params_captions = []

    cocomo_effort_ph = calculate_cocomo_basic(loc_input_val, cocomo_mode_input, eaf_input, HOURS_PER_PERSON_MONTH)
    all_estimation_results['COCOMO II (Basic)'] = cocomo_effort_ph
    traditional_params_captions.append(
        f"* **COCOMO II (Basic):** Mode={cocomo_mode_input}, LOC={loc_input_val}, EAF={eaf_input}")

    fp_effort_ph = calculate_fp_effort(fp_input_val, hours_per_fp_input)
    all_estimation_results['Function Points'] = fp_effort_ph
    traditional_params_captions.append(f"* **Function Points:** FP={fp_input_val}, Hours/FP={hours_per_fp_input}")

    ucp_effort_ph = calculate_ucp_effort(ucp_input_val, hours_per_ucp_input)
    all_estimation_results['Use Case Points'] = ucp_effort_ph
    traditional_params_captions.append(f"* **Use Case Points:** UCP={ucp_input_val}, Hours/UCP={hours_per_ucp_input}")

    if traditional_params_captions:
        st.markdown("**Tham số sử dụng cho mô hình truyền thống:**")
        for caption_text in traditional_params_captions: st.markdown(caption_text)
    st.caption("Lưu ý: Kết quả 'Lỗi' cho mô hình truyền thống xuất hiện nếu đầu vào không hợp lệ.")

    # --- 3. Hiển thị Bảng và Biểu đồ So sánh Tổng hợp ---
    st.markdown("#### 3. Bảng và Biểu đồ So sánh (Person-Hours)")

    if all_estimation_results:
        result_df_list = [{'Mô Hình Ước Tính': name, 'Effort Dự đoán (person-hours)': effort} for name, effort in
                          all_estimation_results.items()]
        result_summary_df = pd.DataFrame(result_df_list)


        def format_effort_for_display(x_val):
            if isinstance(x_val, (int, float)): return f"{x_val:,.2f}"
            return str(x_val)


        st.write("Bảng so sánh kết quả:")
        st.dataframe(
            result_summary_df.style.format({'Effort Dự đoán (person-hours)': format_effort_for_display}),
            use_container_width=True, hide_index=True
        )

        st.write("Biểu đồ so sánh:")
        try:
            chart_df_source = result_summary_df.copy()
            chart_df_source['Effort Dự đoán (person-hours)'] = chart_df_source['Effort Dự đoán (person-hours)'].astype(
                str).str.replace(',', '', regex=False)
            chart_df_source['Effort Dự đoán (person-hours)'] = pd.to_numeric(
                chart_df_source['Effort Dự đoán (person-hours)'], errors='coerce')
            chart_df_source.dropna(subset=['Effort Dự đoán (person-hours)'], inplace=True)

            if not chart_df_source.empty:
                chart_df_source = chart_df_source.sort_values(by='Effort Dự đoán (person-hours)', ascending=False)
                # Sử dụng st.bar_chart() của Streamlit thay vì Matplotlib trực tiếp cho đơn giản và tương thích tốt hơn
                chart_data_for_st = chart_df_source.set_index('Mô Hình Ước Tính')['Effort Dự đoán (person-hours)']
                st.bar_chart(chart_data_for_st)
            else:
                st.info("Không có dự đoán/tính toán hợp lệ (kiểu số) để vẽ biểu đồ so sánh.")
        except Exception as chart_render_e:
            st.warning(f"Không thể vẽ biểu đồ so sánh: {chart_render_e}")
            print(traceback.format_exc())
    else:
        st.warning("Không có kết quả nào để hiển thị.")

    if ml_error_messages:
        st.subheader("⚠️ Chi tiết lỗi dự đoán ML:")
        for model_name_err_disp, msg_err_disp in ml_error_messages.items():
            st.caption(f"**{model_name_err_disp}:** {msg_err_disp}")

    st.info("""
    **Lưu ý quan trọng:** Kết quả chỉ là ước tính. Effort thực tế có thể khác biệt.
    Độ chính xác của ML phụ thuộc vào dữ liệu huấn luyện và preprocessor.
    Độ chính xác của mô hình truyền thống phụ thuộc vào việc chọn đúng tham số.
    """)

# Xử lý trường hợp không tải được artifacts ban đầu (hiển thị thông báo nếu nút chưa được nhấn VÀ artifacts lỗi)
elif not calculate_button and not artifacts_load_status:
    st.error("Không thể tải các thành phần ML cần thiết. Phần dự đoán ML sẽ không hoạt động.")
    st.info(f"Kiểm tra các file .joblib trong thư mục OUTPUT_DIR (hiện tại: '{os.path.abspath(OUTPUT_DIR)}').")
    st.info("Bạn vẫn có thể sử dụng tính toán từ mô hình truyền thống.")

# --- Chân trang ---
st.markdown("---")
st.caption(f"Ứng dụng demo. Các file artifacts được tìm kiếm trong: {os.path.abspath(OUTPUT_DIR)}")

