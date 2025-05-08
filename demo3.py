# -*- coding: utf-8 -*-
"""
app.py: Ứng dụng Web Streamlit để dự đoán Effort
(Bao gồm Mô hình ML, COCOMO II Basic, FP, UCP và So sánh).

(Phiên bản đã sửa lỗi "Unknown Categories", cập nhật đường dẫn file,
và sửa lỗi hiển thị biểu đồ)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import OrderedDict
import math # Cần cho COCOMO
import traceback # Thêm để in lỗi chi tiết

# Import các lớp cần thiết (Giữ nguyên như cũ)
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder # Hoặc StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
except ImportError as e:
    st.error(f"Lỗi Import thư viện: {e}. Hãy đảm bảo các thư viện cần thiết (streamlit, pandas, scikit-learn, joblib, xgboost) đã được cài đặt trong môi trường của bạn.")
    st.stop()

# Add this function after your imports but before the UI code
def convert_metrics(loc=None, fp=None, ucp=None, language_type="3GL"):
    """Convert between LOC, FP, and UCP based on provided value and language type."""
    # Conversion ratios based on language type
    conversion_ratios = {
        "3GL": {"loc_per_fp": 125, "ucp_to_fp": 1.3},
        "4GL": {"loc_per_fp": 40, "ucp_to_fp": 1.3}
    }

    ratio = conversion_ratios[language_type]

    results = {}

    if loc is not None and loc > 0:
        # LOC to FP and UCP
        results["fp"] = round(loc / ratio["loc_per_fp"], 2)
        results["ucp"] = round(results["fp"] / ratio["ucp_to_fp"], 2)
        results["loc"] = loc
    elif fp is not None and fp > 0:
        # FP to LOC and UCP
        results["loc"] = round(fp * ratio["loc_per_fp"], 0)
        results["ucp"] = round(fp / ratio["ucp_to_fp"], 2)
        results["fp"] = fp
    elif ucp is not None and ucp > 0:
        # UCP to FP and LOC
        results["fp"] = round(ucp * ratio["ucp_to_fp"], 2)
        results["loc"] = round(results["fp"] * ratio["loc_per_fp"], 0)
        results["ucp"] = ucp

    # Estimate development time and team size
    if "loc" in results:
        # Rough heuristics for team size based on project size
        if results["loc"] < 10000:
            results["team_size"] = 3
        elif results["loc"] < 50000:
            results["team_size"] = 5
        else:
            results["team_size"] = 8

        # Rough dev time estimate in months (assuming 152 hrs/month)
        if "fp" in results:
            effort_hours = results["fp"] * 10  # Using 10 hours/FP as example productivity
            results["dev_time_months"] = round(effort_hours / (results["team_size"] * 152), 1)

    return results


# --- Tạo Giao diện Nhập liệu ---
st.sidebar.header("Nhập Thông tin Dự án")
input_values = {}  # Dictionary chung để lưu tất cả giá trị người dùng nhập

# --- Thêm phần chuyển đổi tự động ---
st.sidebar.subheader("Chuyển đổi Đo lường")
language_type = st.sidebar.selectbox("Loại Ngôn ngữ Lập trình", ["3GL", "4GL"],
                                     help="3GL: Java, C++, v.v. 4GL: Python, SQL, v.v.")

# Bật/tắt tính năng chuyển đổi tự động
auto_convert = st.sidebar.checkbox("Bật Chuyển đổi Tự động", value=True,
                                   help="Tự động chuyển đổi giữa các đo lường khi một giá trị thay đổi")

if auto_convert:
    # Lưu trữ loại đo lường đang được chỉnh sửa
    if 'metric_being_edited' not in st.session_state:
        st.session_state.metric_being_edited = None


    # Hàm callback để đánh dấu đo lường đang được chỉnh sửa
    def set_loc():
        st.session_state.metric_being_edited = 'loc'


    def set_fp():
        st.session_state.metric_being_edited = 'fp'


    def set_ucp():
        st.session_state.metric_being_edited = 'ucp'


    # Tạo 3 cột cho input LOC, FP, UCP
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        loc_value = st.number_input("Lines of Code (LOC)", min_value=0,
                                    value=int(input_values.get('LOC', 10000)),
                                    step=100, on_change=set_loc, key="loc_conv")

    with col2:
        fp_value = st.number_input("Function Points (FP)", min_value=0.0,
                                   value=float(input_values.get('FP', 100.0)),
                                   step=10.0, format="%.2f", on_change=set_fp, key="fp_conv")

    with col3:
        ucp_value = st.number_input("Use Case Points (UCP)", min_value=0.0,
                                    value=float(input_values.get('UCP', 100.0)),
                                    step=10.0, format="%.2f", on_change=set_ucp, key="ucp_conv")

    # Thực hiện chuyển đổi dựa trên đo lường đang được chỉnh sửa
    if st.session_state.metric_being_edited == 'loc' and loc_value > 0:
        results = convert_metrics(loc=loc_value, language_type=language_type)
        if results:
            # Cập nhật các trường nhập liệu
            if 'fp_conv' in st.session_state and 'ucp_conv' in st.session_state:
                st.session_state.fp_conv = results.get('fp', 0)
                st.session_state.ucp_conv = results.get('ucp', 0)
            # Cập nhật input_values để sử dụng trong tính toán
            input_values['LOC'] = results.get('loc', 0)
            input_values['FP'] = results.get('fp', 0)
            input_values['UCP'] = results.get('ucp', 0)
            # Cập nhật thời gian phát triển và kích thước nhóm nếu có
            if 'Development Time (months)' in original_cols_order:
                input_values['Development Time (months)'] = results.get('dev_time_months', 6)
            if 'Team Size' in original_cols_order:
                input_values['Team Size'] = results.get('team_size', 5)

    elif st.session_state.metric_being_edited == 'fp' and fp_value > 0:
        results = convert_metrics(fp=fp_value, language_type=language_type)
        if results:
            # Cập nhật các trường nhập liệu
            if 'loc_conv' in st.session_state and 'ucp_conv' in st.session_state:
                st.session_state.loc_conv = results.get('loc', 0)
                st.session_state.ucp_conv = results.get('ucp', 0)
            # Cập nhật input_values để sử dụng trong tính toán
            input_values['LOC'] = results.get('loc', 0)
            input_values['FP'] = results.get('fp', 0)
            input_values['UCP'] = results.get('ucp', 0)
            # Cập nhật thời gian phát triển và kích thước nhóm nếu có
            if 'Development Time (months)' in original_cols_order:
                input_values['Development Time (months)'] = results.get('dev_time_months', 6)
            if 'Team Size' in original_cols_order:
                input_values['Team Size'] = results.get('team_size', 5)

    elif st.session_state.metric_being_edited == 'ucp' and ucp_value > 0:
        results = convert_metrics(ucp=ucp_value, language_type=language_type)
        if results:
            # Cập nhật các trường nhập liệu
            if 'loc_conv' in st.session_state and 'fp_conv' in st.session_state:
                st.session_state.loc_conv = results.get('loc', 0)
                st.session_state.fp_conv = results.get('fp', 0)
            # Cập nhật input_values để sử dụng trong tính toán
            input_values['LOC'] = results.get('loc', 0)
            input_values['FP'] = results.get('fp', 0)
            input_values['UCP'] = results.get('ucp', 0)
            # Cập nhật thời gian phát triển và kích thước nhóm nếu có
            if 'Development Time (months)' in original_cols_order:
                input_values['Development Time (months)'] = results.get('dev_time_months', 6)
            if 'Team Size' in original_cols_order:
                input_values['Team Size'] = results.get('team_size', 5)
else:
    # Nếu không bật chế độ tự động, sử dụng giao diện nhập liệu tiêu chuẩn
    # Nhóm các input cần cho cả ML và truyền thống lại với nhau
    st.sidebar.subheader("Đặc trưng Cơ bản (Sử dụng bởi nhiều mô hình)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        # LOC cần cho ML (nếu có) và COCOMO
        if 'LOC' in original_cols_order or True:  # Luôn hiển thị LOC vì cần cho COCOMO
            input_values['LOC'] = st.number_input("Lines of Code (LOC)", min_value=0, value=10000, step=100,
                                                  key="loc_input")
        # FP cần cho ML (nếu có) và FP Estimation
        if 'FP' in original_cols_order or True:  # Luôn hiển thị FP
            input_values['FP'] = st.number_input("Function Points (FP)", min_value=0, value=100, step=10,
                                                 key="fp_input")
    with col2:
        # UCP cần cho ML (nếu có) và UCP Estimation
        if 'UCP' in original_cols_order or True:  # Luôn hiển thị UCP
            input_values['UCP'] = st.number_input("Use Case Points (UCP)", min_value=0.0, value=100.0, step=10.0,
                                                  format="%.2f", key="ucp_input")
