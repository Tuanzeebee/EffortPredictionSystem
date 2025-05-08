import pandas as pd
import numpy as np
import random

# Thiết lập seed để tái lập
random.seed(42)
np.random.seed(42)

# Danh sách các giá trị hợp lệ
project_types = ['Web', 'Mobile', 'Enterprise', 'Embedded']
language_types = ['3GL', '4GL']
primary_languages = {
    '3GL': ['C', 'C#', 'C++', 'Java', '.Net', 'ASP.Net', 'PHP', 'Javascript', 'Python', 'Visual Basic'],
    '4GL': ['SQL', 'Oracle Forms', 'ABAP', 'PowerBuilder']
}
sloc_per_fp_map = {
    'C': 148, 'Basic': 107, 'C#': 59, 'C++': 60, 'Java': 60,
    'Visual Basic': 50, 'HTML4': 14, 'SQL': 13, '.Net': 80,
    'Oracle Forms': 20, 'ABAP': 20, 'PowerBuilder': 20,
    'PHP': 80, 'Javascript': 80, 'Python': 80, 'ASP.Net': 80, 'ASP': 80
}
count_approaches = ['IFPUG 4+', 'COSMIC', 'FiSMA', 'NESMA']
application_groups = ['Business Application', 'Infrastructure Software', 'Mathematically Intensive Application', 'Real-Time Application']
application_types = ['E-Business', 'Air Traffic Management', 'Transaction System', 'IT Management', 'Website', 'Human Resources', 'Internet Banking']
development_types = ['New Development', 'Enhancement', 'Re-development', 'Not Defined', 'Other']

def generate_record(project_id):
    project_type = random.choice(project_types)

    # Gán language type dựa vào project type
    lang_type = '4GL' if project_type == 'Enterprise' and random.random() < 0.6 else '3GL'
    if lang_type == '4GL':
        primary_lang = random.choice(primary_languages['4GL'])
    else:
        primary_lang = random.choice(primary_languages['3GL'])

    sloc_per_fp = sloc_per_fp_map.get(primary_lang, 80)
    fp = np.random.randint(100, 1000)

    # UCP theo loại dự án
    if project_type == 'Web':
        ucp = round(fp * np.random.uniform(0.9, 1.0), 2)
    elif project_type == 'Enterprise':
        ucp = round(fp * np.random.uniform(1.0, 1.1), 2)
    else:
        ucp = round(fp * np.random.uniform(0.9, 1.1), 2)

    loc = round(fp * sloc_per_fp / 1000, 2)

    # Team size và thời gian
    if project_type in ['Embedded', 'Mobile'] or 'Real-Time' in application_groups:
        team_size = np.random.randint(2, 6)
    else:
        team_size = np.random.randint(3, 15)

    dev_time = np.random.randint(3, 19)
    effort = team_size * dev_time * 160

    count_approach = random.choice(count_approaches)
    app_group = random.choice(application_groups)
    app_type = random.choice(application_types)
    dev_type = random.choice(development_types)

    # Defect Density
    if lang_type == '4GL' or app_group == 'Business Application':
        defect_density = round(np.random.uniform(0.2, 1.5), 2)
    else:
        defect_density = round(np.random.uniform(1.0, 5.0), 2)

    # Productivity (FP / person-month)
    person_month = effort / 160
    productivity = round(fp / person_month, 3)

    return {
        'Project ID': project_id,
        'LOC': loc,
        'FP': fp,
        'UCP': ucp,
        'Effort (person-hours)': effort,
        'Development Time (months)': dev_time,
        'Team Size': team_size,
        'Project Type': project_type,
        'Language Type': lang_type,
        'Primary Programming Language': primary_lang,
        'Count Approach': count_approach,
        'Application Group': app_group,
        'Application Type': app_type,
        'Development Type': dev_type,
        'Defect Density': defect_density,
        'Productivity': productivity,
        'Schedule': dev_time
    }

# Tạo 10000 bản ghi
data = [generate_record(i+1) for i in range(10000)]
df = pd.DataFrame(data)

# Xuất ra file CSV
output_path = "synthetic_software_projects.csv"
df.to_csv(output_path, index=False)
output_path
