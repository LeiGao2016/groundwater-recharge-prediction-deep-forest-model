import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import datetime
start = datetime.datetime.now()
from BODF.Select_K.DeepForest.gc import *    #导入深度森林

# 导入数据集
dataset = pd.read_csv(r'C:\Users\ylhpc\Desktop\test_2.csv') # 请将路径换为自己本地存放源数据的路径
dataset.iloc[:,-4].fillna(method='bfill',inplace = True)    # 对缺失值较多的进行填充
dataset.iloc[:,-5].fillna(method='ffill',inplace = True)    # 对非数值项进行填充
dataset.fillna(dataset.median()['LAI':'clay_depth_avg'],inplace=True)
surface_geol = dataset.iloc[:,-5]
encode = pd.get_dummies(surface_geol)
dataset.iloc[:,-5] = encode
dataset = dataset.sort_values(by='R-ground recharge rate',ascending=True)

# 需要对x进行插值，补充缺失数据
x = pd.DataFrame(dataset.iloc[:,:-3])
y = pd.DataFrame(dataset.iloc[:,-3])
Lati = np.array(pd.DataFrame(dataset.iloc[:,-2]))
Longi = np.array(pd.DataFrame(dataset.iloc[:,-1]))

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaled = scaler_x.fit_transform(x)
x = scaled.astype('float32')

# # 对Y进行归一化处理
from sklearn.preprocessing import MinMaxScaler
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled = scaler_y.fit_transform(y)
# 保证为float ensure all data is float
y = scaled.astype('float32')

# 直接进行预测
x_train, x_left, y_train, y_left = train_test_split(x, y, test_size=0.2, random_state=55)
x_val,x_test,y_val,y_test = train_test_split(x_left,y_left,test_size=0.5,random_state=25)
Lati_train,Lati_left,Longi_train,Longi_left = train_test_split(Lati,Longi,test_size=0.2,random_state=55)
Lati_val,Lati_test,Longi_val,Longi_test = train_test_split(Lati_left,Longi_left,test_size=0.5,random_state=25)

clf = gcForest(num_forests=8,train_data=x_train,train_label=y_train,val_data=x_val,val_label=y_val)
clf.train_and_predict(2)
print("测试集的结果：")
ny_predict = clf.newpredict_data(x_test)
ny_predict = ny_predict.reshape((y_test.shape[0], 1))
npredict_r_2 = r2_score(y_test, ny_predict)
npredict_rmse = np.sqrt(mean_squared_error(y_test, ny_predict))
npredict_mae = mean_absolute_error(y_test, ny_predict)
print('r2为：', npredict_r_2, ",rmse为：", npredict_rmse, ",mae为：", npredict_mae)

print("训练集的结果：")
ny_train_predict = clf.newpredict_data(x_train)
ny_train_predict = ny_train_predict.reshape((y_train.shape[0], 1))
ntrain_predict_r_2 = r2_score(y_train, ny_train_predict)
ntrain_predict_rmse = np.sqrt(mean_squared_error(y_train, ny_train_predict))
ntrain_predict_mae = mean_absolute_error(y_train, ny_train_predict)
print('r2为：', ntrain_predict_r_2, ",rmse为：", ntrain_predict_rmse, ",mae为：", ntrain_predict_mae)

print("验证集的结果：")
ny_val_predict = clf.newpredict_data(x_val)
ny_val_predict = ny_val_predict.reshape((y_val.shape[0], 1))
nval_predict_r_2 = r2_score(y_val, ny_val_predict)
nval_predict_rmse = np.sqrt(mean_squared_error(y_val, ny_val_predict))
nval_predict_mae = mean_absolute_error(y_val, ny_val_predict)
print('r2为：', nval_predict_r_2, ",rmse为：", nval_predict_rmse, ",mae为：", nval_predict_mae)


end = datetime.datetime.now()
print('运行时间{}秒'.format(end-start))
