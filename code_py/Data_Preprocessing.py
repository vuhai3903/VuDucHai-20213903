import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def ml():
   
  
    df4 = pd.read_csv(r"C:\python\dataset\capture20110815-2.csv")
    df5= pd.read_csv(r"C:\python\dataset\capture20110815-3.csv")
    df11 = pd.read_csv(r"C:\python\dataset\capture20110818.csv")
    df12 = pd.read_csv(r"C:\python\dataset\capture20110818-2.csv")

    
    df = pd.concat([df4,df5,df12,df11], ignore_index=True)
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
    df = df.fillna(10)



# 2 Mã hóa cột 'Label' thành số nguyên
   

#2.1 mã hoá với giá trị số
    # def encode_dir_proto(df):
    #     dir_map = {'<->': 1, '->': 2}
    #     proto_map = {'udp': 20, 'tcp': 30, 'icmp': 40}

    #     df['Dir'] = df['Dir'].str.strip() 
    #     df['Dir'] = df['Dir'].map(dir_map).fillna(0)   

    #     df['Proto'] = df['Proto'].str.strip()
    #     df['Proto'] = df['Proto'].map(proto_map).fillna(0)

    #     return df

    # df = encode_dir_proto(df)
 

#  2.2 one hot endcoding

    def OneHot_Ending(df):

        dir_values = ['<->', '->']
        proto_values = ['tcp', 'udp', 'icmp']
        tos_values = ['0.0', '10.0']

        df['Dir'] = df['Dir'].str.strip().apply(lambda x: x if x in dir_values else 'others')
        df['Proto'] = df['Proto'].str.lower().apply(lambda x: x if x in proto_values else 'others')
        df['sTos'] = df['sTos'].apply(lambda x: x if x in tos_values else 'others')
        df['dTos'] = df['dTos'].apply(lambda x: x if x in tos_values else 'others')

        df_encoded = pd.get_dummies(df, columns=['Dir', 'Proto', 'sTos', 'dTos'])

        return df_encoded

    df = OneHot_Ending(df)
  
    x = df.drop('Label', axis=1)
    y = df['Label']    # do chàm get_dummies chèn thê cột mới như dir__other ,..  sau cột label nên phải chia x, y như này
   

  
   
#  3.1 anov

#     # def anova(x, y):
    
#     #     x1 = SelectKBest(score_func=f_classif,k=10)  # k là cọt đặc trưng muốn giữ 
#     #     x_new = x1.fit_transform(x, y)
#     #     return x_new
    
#     # x = anova(x, y)
  


#3.2 chi-squared
    

    def chi_squared(x, y):

        x1 = SelectKBest(chi2, k=10)
        x_new = x1.fit_transform(x, y)

        return x_new
    
    x = chi_squared(x,y)



# # #4.1 RandomUnderSampler 



# #     def RandomUndersampler(x, y):
    
# #         rus = RandomUnderSampler(random_state=0, sampling_strategy={0: 240000, 1: 158000, 2: 80451})
# #         x, y = rus.fit_resample(x, y)
        

# #         return x, y
# #     x , y = RandomUndersampler(x,y)


 
#4.2 near-miss 1
    

    def Nearmiss(x, y):
        nm = NearMiss(version=1, sampling_strategy={0: 240000, 1: 158000, 2: 80451}) # tổng 0 = 1 +2 
        x, y = nm.fit_resample(x, y)

        
        return x,y
    x, y = Nearmiss(x,y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# bước 5.1  chuẩn hóa Normalization 
#     # def Normalization(x_train ,x_test):
    
#     #     scaler = MinMaxScaler(feature_range=(0,1)).fit(x_train)

#     #     x_train = scaler.transform(x_train)
#     #     x_test = scaler.transform(x_test)
#     #     return x_train ,x_test
#     # x_train , x_test = Normalization rmalize_data (x_train, x_test)

# #5.2 chuẩn hóa Standardization  
    # def Standardization(x_train,x_test):
    
    #     scaler = StandardScaler().fit(x_train)
    #     x_train = scaler.transform(x_train)
    #     x_test = scaler.transform(x_test)
    #     return x_train ,x_test
    # x_train , x_test = Standardization (x_train, x_test)
        
 
    return x_train, x_test, y_train, y_test
ml()