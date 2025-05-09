import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def ml():
    import numpy as np
    import pandas as pd
    df3 = pd.read_csv(r"C:\python\dataset\capture20110815-2.csv")
    df4 = pd.read_csv(r"C:\python\dataset\capture20110815-3.csv")
    df5 = pd.read_csv(r"C:\python\dataset\capture20110815.csv")
    df10 = pd.read_csv(r"C:\python\dataset\capture20110818-2.csv")
    df12= pd.read_csv(r"C:\python\dataset\capture20110818.csv")
    
 

    # Nối 2 DataFrame lại theo chiều dọc (nhiều dòng hơn)
    df = pd.concat([df3,df4,df5,df10,df12], ignore_index=True)
    def map_flow(label):
        if 'Botnet' in label:
            return 1 
        elif 'Normal' in label:
            return 2 
        else:
            return 0
    df['Label'] = df['Label'].apply(map_flow)
    
    # Xóa các cột không cần thiết
    drop = ['StartTime', 'SrcAddr', 'Sport', 'DstAddr', 'Dport', 'State']
    df = df.drop(drop, axis=1)

   
    #Xử lý giá trị thiếu và vô cùng
    for col in df.columns[:-1]:  # Duyệt qua tất cả cột trừ cột cuối (Label)
        missing = df[col].isnull().sum()
        # inf_mask = df[col] == np.inf
        # neginf_mask = df[col] == -np.inf

        if missing  > 0:
            # df.loc[inf_mask, col] = np.nan
            # df.loc[neginf_mask, col] = np.nan

            imputer = SimpleImputer(strategy='mean')  
            df[[col]] = imputer.fit_transform(df[[col]])

    # Mã hóa cột 'Label' thành số nguyên
   

# #2.1 mã hoá với giá trị số

#     def encode_dir_proto(df):
#         dir_map = {'<->': 1, '->': 2}
#         proto_map = {'udp': 17, 'tcp': 6, 'icmp': 1}

#         df['Dir'] = df['Dir'].astype(str).str.strip()  # <-- bỏ khoảng trắng đầu/cuối
#         df['Dir'] = df['Dir'].map(dir_map).fillna(0)   # ánh xạ sau khi làm sạch

#         df['Proto'] = df['Proto'].astype(str).str.strip()
#         df['Proto'] = df['Proto'].map(proto_map).fillna(0)

#         return df


#     df = encode_dir_proto(df)
 

#   #  2.2 one hot endcoding


# # 2.2 one hot encoding
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    

    def encode_data(df):
    
        
        dir_values = ['<->', '->']
        proto_values = ['tcp', 'udp', 'icmp']
        sTos_values = ['0.0', '3.0', '192.0']  
        dTos_values = ['0.0', '3.0', '2.0']          

        def map_limited_values(val, valid_list):
            return val if val in valid_list else 'others'

        df['Dir'] = df['Dir'].str.strip().apply(lambda x: x if x in dir_values else 'others')
        df['Proto'] = df['Proto'].apply(lambda x: map_limited_values(x.lower(), proto_values)) 
        df['sTos'] = df['sTos'].astype(str).apply(lambda x: map_limited_values(x, sTos_values))
        df['dTos'] = df['dTos'].astype(str).apply(lambda x: map_limited_values(x, dTos_values))
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        categorical_cols = ['Dir', 'Proto', 'sTos', 'dTos']
        encoded = encoder.fit_transform(df[categorical_cols])

        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

        df_final = df.drop(columns=categorical_cols).reset_index(drop=True)
        df_final = pd.concat([df_final, encoded_df.reset_index(drop=True)], axis=1)


        return df_final

        # Xử lý và mã hóa dữ liệu
    df = encode_data(df)

  
    x = df.drop('Label', axis=1)
    y = df['Label']    # do chàm get_dummies chèn thê cột mới như dir__other ,..  sau cột label nên phải chia x, y như này
   

  
   
#   #  3.1 anov
    # from sklearn.feature_selection import SelectKBest, f_classif

    # def select_features_anova(x, y):
    
    #     selector = SelectKBest(score_func=f_classif,k=10)  # k là cọt đặc trưng muốn giữ 
    #     x_new = selector.fit_transform(x, y)
    #     return x_new
    
    # x = select_features_anova(x, y)
  


    #3.2 chi-squared
    from sklearn.feature_selection import SelectKBest, chi2

    def chay_chi_squared(x, y):

        # scaler = MinMaxScaler()
        # x_scaled = scaler.fit_transform(x)

        # Sau đó mới chạy chi2
        x1 = SelectKBest(chi2, k=12)
        x=x1.fit_transform(x, y)

        return x
    x = chay_chi_squared(x,y)





# #4.1 RandomUnderSampler + TomekLinks
#     from imblearn.under_sampling import RandomUnderSampler
#     from imblearn.under_sampling import TomekLinks

#     from sklearn.datasets import make_classification
#     import numpy as np

#     def random_undersampler_tomeklinks(x, y):
    
#         # Bước 1: Random Under Sampling
#         rus = RandomUnderSampler(random_state=0, sampling_strategy={0: 240000, 1: 158000, 2: 80451})
#         x_resampled, y_resampled = rus.fit_resample(x, y)
        
        # # Bước 2: Áp dụng TomekLinks
       # # tl = TomekLinks()
       # # x_resampled_tomek, y_resampled_tomek = tl.fit_resample(x_resampled, y_resampled)
        
#         return x_resampled, y_resampled
#     x , y = random_undersampler_tomeklinks (x,y)
    
#     # x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)
#     x, y = random_undersampler_tomeklinks(x, y)


 
#4.2 near-miss 1
    from imblearn.under_sampling import NearMiss
    import numpy as np

    def near_miss_then_smote(x, y):
        nm = NearMiss(version=1, sampling_strategy={0: 240000, 1: 158000, 2: 80451}) # tổng 0 = 1 +2 
        x_resampled, y_resampled = nm.fit_resample(x, y)

        return x_resampled,y_resampled
    x, y = near_miss_then_smote(x,y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

#    # bước 5.1  chuẩn hóa Normalization 
    # def normalize_data(x_train ,x_test):
    
    #     scaler = MinMaxScaler(feature_range=(0,1)).fit(x_train)

    #     x_train = scaler.transform(x_train)
    #     x_test = scaler.transform(x_test)
    #     return x_train ,x_test
    # x_train , x_test = normalize_data (x_train, x_test)
    # #5.2 chuẩn hóa Standardization  
    # def standardize_data(x_train,x_test):
    
    #     scaler = StandardScaler().fit(x_train)
    #     x_train = scaler.transform(x_train)
    #     x_test = scaler.transform(x_test)
    #     return x_train ,x_test
    # x_train , x_test = standardize_data (x_train, x_test)
        
 
    return x_train, x_test, y_train, y_test
ml()