import pandas as pd
import joblib
from datetime import datetime
import subprocess
import numpy as np
import os

try :
    df = pd.read_csv(r'C:\python\dataset\capture20110816-3.csv').head(50000)
    #Xóa các cột không cần thiết
    drop = ['StartTime', 'SrcAddr', 'Sport', 'DstAddr', 'Dport', 'State']
    df = df.drop(drop, axis=1)
    df = df.fillna(10)

    # One-hot encoding
    df_test = pd.get_dummies(df, columns=['Dir', 'Proto', 'dTos'])


    processed_columns = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes',  # danh sách cột train
                        'Dir_->', 'Dir_<->',
                        'Proto_icmp','Proto_tcp', 'Proto_udp',
                        'dTos_0.0', 'dTos_10.0']


    df_test.columns = df_test.columns.str.replace(' ', '', regex=False)

    # Duyệt qua các cột và gom các one-hot "lạ" vào *_others
    for col in df_test.columns:
        if col.startswith('Dir_') and col not in processed_columns:
    
            df_test.drop(columns=col, inplace=True)
        elif col.startswith('dTos_') and col not in processed_columns:
            df_test.drop(columns=col, inplace=True)

    # Thêm các cột còn thiếu (nếu có) với giá trị = 0
    for col in processed_columns:
        if col not in df_test.columns:
            df_test[col] = 0

    df_test = df_test[processed_columns]


    model = joblib.load(r'C:\python\VuHai-20213903\code_py\dump_random_forest.pkl')


    y_pred = model.predict(df_test.values)

    df_test['y_pred'] = y_pred     


    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)

    count_botnet = 0
    percent_botnet = 0
    for label, count in zip(unique, counts):
        percent_botnet = (count / total) * 100
        print(f"Giá trị {label}: {count} mẫu ({percent_botnet:.2f}%)")
    
        if label == 1: 
            count_botnet = count
            if percent_botnet > 9 and count_botnet > 1000:
                
                attack_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        

      #       Gọi file ml.py
                subprocess.run(['python', 'ml.py'])
        
                rows_label_1 = df_test[df_test['y_pred'] == 1]  # in ra các dòng label = 1 
                rows_label_1 = rows_label_1.drop("y_pred", axis=1)

                dur = round( rows_label_1['Dur'].mean() , 3 ) 
                tot_pkts = int( rows_label_1['TotPkts'].mean())
                tot_bytes = int( rows_label_1['TotBytes'].mean() )
                src_bytes = int( rows_label_1['SrcBytes'].mean() )
                dtos_0_0 = round ( rows_label_1['dTos_0.0'].mean() ,3 )
                dtos_10_0 = round ( rows_label_1['dTos_10.0'].mean() ,3 )


                # Đếm số lượng True trong mỗi cột dir
                count_dir1 = rows_label_1['Dir_->'].sum()  # Số lượng True trong cột Dir_->
                count_dir2 = rows_label_1['Dir_<->'].sum()  # Số lượng True trong cột Dir_<->
                if count_dir1 > count_dir2  :
                    
                    dir_forward = 1
                    dir_bidirectional = 0
                
                    
                elif count_dir2 > count_dir1 :
                    
                    dir_forward = 0
                    dir_bidirectional = 1
                
                else:  # các trường hợp khác không thể dự đoán được nên cho bằng NaN hết    
                    dir_forward = None
                    dir_bidirectional = None


                # Đếm số lượng True trong mỗi cột proto
                count_icmp = rows_label_1['Proto_icmp'].sum()  # Số lượng True trong cột Proto_icmp
                count_tcp = rows_label_1['Proto_tcp'].sum()  # Số lượng True trong cột Proto_tcp
                count_udp = rows_label_1['Proto_udp'].sum()  # Số lượng True trong cột Proto_udp

                if count_icmp > count_tcp and count_icmp > count_udp :
                    proto_icmp = 1
                    proto_tcp = 0
                    proto_udp = 0

                elif count_tcp > count_icmp and count_tcp > count_udp :
                    proto_icmp = 0
                    proto_tcp = 1
                    proto_udp = 0
                    
                elif count_udp > count_icmp and count_udp > count_tcp :
                    proto_icmp = 0
                    proto_tcp = 0
                    proto_udp = 1
                    
                else:# các trường hợp khác không thể dự đoán được nên cho bằng NaN hết    
                    proto_icmp = None
                    proto_tcp = None
                    proto_udp = None


except Exception as e : 
    print (f'Lỗi {e} , xin vui lòng thử lại !')
    
