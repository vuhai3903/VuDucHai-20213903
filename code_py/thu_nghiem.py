import pandas as pd
import joblib
from datetime import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
# Đọc dữ liệu
df = pd.read_csv(r'C:\python\dataset\sss.csv')
# df = df[df['Label'].str.contains('Normal', case=False, na=False)]

#Xóa các cột không cần thiết
drop = ['StartTime', 'SrcAddr', 'Sport', 'DstAddr', 'Dport', 'State']
df = df.drop(drop, axis=1)

# One-hot encoding
df_test = pd.get_dummies(df, columns=['Dir', 'Proto', 'dTos'])

# Xóa dấu cách khỏi tên cột
df_test.columns = df_test.columns.str.replace(' ', '', regex=False)

# Các cột cần giữ lại (giống lúc train model)
processed_columns = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes', 
                     'Dir_->', 'Dir_<->', 'Dir_others', 
                     'Proto_icmp', 'Proto_tcp', 'Proto_udp',
                     'dTos_0.0', 'dTos_others']

# Khởi tạo các cột *_others nếu chưa có
for col in ['Dir_others',  'dTos_others']:
    if col not in df_test.columns:
        df_test[col] = 0

# Duyệt qua các cột và gom các one-hot "lạ" vào *_others
for col in df_test.columns:
    if col.startswith('Dir_') and col not in processed_columns:
        df_test['Dir_others'] += df_test[col]
        df_test.drop(columns=col, inplace=True)
    elif col.startswith('dTos_') and col not in processed_columns:
        df_test['dTos_others'] += df_test[col]
        df_test.drop(columns=col, inplace=True)

# Thêm các cột còn thiếu (nếu có) với giá trị = 0
for col in processed_columns:
    if col not in df_test.columns:
        df_test[col] = 0

df_test = df_test[processed_columns]


model = joblib.load('dump_random_forest.pkl')


y_pred = model.predict(df_test)


df_test['y_pred'] = y_pred     
rows_label_1 = df_test[df_test['y_pred'] == 1]  # in ra các dòng label = 1 trừ cột 
rows_label_1 = rows_label_1.drop("y_pred", axis=1)


def detect_dos_attack_type(rows_label_1):  # phân biệt các loại tấn công tcp flood , udp flood , icmp flood và other
    total = len(rows_label_1)
    present_tcp = present_udp = present_icmp = 0
  

    for _, row in rows_label_1.iterrows():
        proto_tcp = row.get('Proto_tcp', 0)
        proto_udp = row.get('Proto_udp', 0)
        proto_icmp = row.get('Proto_icmp', 0)
        dur = row['Dur']
        src_bytes = row['SrcBytes']
        tot_pkts = row['TotPkts']
        tot_bytes = row['TotBytes']

        if proto_tcp == 1 and src_bytes <= 200 and dur < 3 and tot_pkts > 4 and tot_bytes > 100:
            present_tcp += 1

        elif proto_udp == 1 and tot_pkts > 1 and tot_bytes > 100 and src_bytes < 200:
            present_udp += 1

        elif proto_icmp == 1 and tot_bytes > 65000:
            present_icmp += 1

    # Sau vòng lặp mới xác định loại tấn công
    if present_tcp / total >= 0.7:
        attack_types = 'TCP Flood'
    elif present_udp / total >= 0.7:
        attack_types = 'UDP Flood'
    elif present_icmp / total >= 0.7:
        attack_types = 'ICMP Flood'
    else:
        attack_types = 'DoS_others'

    return attack_types

attack_types = detect_dos_attack_type(rows_label_1)

print (attack_types)


# unique, counts = np.unique(y_pred, return_counts=True)
# total = len(y_pred)


# for label, count in zip(unique, counts):
#     percent = (count / total) * 100
#     print(f"Giá trị {label}: {count} mẫu ({percent:.2f}%)")
    
# percent_botnet = 0
# if 1 in unique:
#     index = list(unique).index(1)
#     percent_botnet = (counts[index] / total) * 100
    
# df_test['y_pred'] = y_pred     
# rows_botnet = df_test[df_test['y_pred'] == 1]  # các dòng label = 1 


# if percent_botnet > 50 and counts[index] > 1000:  #counts[index] số mẫu 1 ( tấn công )
#     attack_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


#     print("Attack detected!")
#     print(f"Time: {attack_time}")

    
#     log = f"{attack_time},{counts[index]}\n"
#     with open("attack_log.csv", "a") as file:
#         file.write(log)
        
#     # Gọi file ml.py
#     # subprocess.run(['python', 'ml.py'])
