import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
plt.rcParams['font.family']='Microsoft YaHei' #显示中文标签
plt.style.use ('ggplot') # 设定绘图风格
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings("ignore")
import datetime
start = datetime.datetime.now()
from BODF.Select_K.BoDeepForest.Comparegc import *    #导入深度森林

# 导入数据集
dataset = pd.read_csv(r'C:\Users\ylhpc\Desktop\test_2.csv')  # 请将路径换为自己本地存放源数据的路径
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

indices=[ 0,13,12,5,11,9,4,1,7,6,14,3,2,10,8]
new_x = np.zeros([x.shape[0],len(indices)-5])
for i in range(len(indices)-5):
    new_x[:,i] = x[:,indices[i]]
x,y = new_x,y


print("开始进行五折交叉验证")
kf = KFold(5, True, random_state=25)
DF = []
r2_list,mae_list,rmse_list = [],[],[]
for train_index,test_index in kf.split(x):
    train_x,test_x = x[train_index,:],x[test_index,:]
    train_y,test_y = y[train_index],y[test_index]
    train_Lati,test_Lati = Lati[train_index,:],Lati[test_index,:]
    train_Longi,test_Longi = Longi[train_index,:],Longi[test_index,:]

    val_x,new_test_x,val_y,new_test_y = train_test_split(test_x,test_y,test_size=0.5,random_state=25)

    clf = gcForest(num_forests=8, train_data=train_x,train_label=train_y,val_data=val_x,val_label=val_y)
    clf.train_and_predict(2)

    y_predict = clf.predict_data(test_x)
    y_predict = y_predict.reshape((test_y.shape[0],1))
    predict_r_2 = r2_score(test_y,y_predict)
    predict_mae = mean_absolute_error(test_y,y_predict)
    predict_rmse = np.sqrt(mean_squared_error(test_y,y_predict))

    y_predict = scaler_y.inverse_transform(y_predict)
    dt = {"R_Ground":y_predict[:,0],"Latitude":test_Lati[:,0],"Longitude":test_Longi[:,0]}
    df = pd.DataFrame(dt,columns=["R_Ground","Latitude","Longitude"])

    r2_list.append(predict_r_2)
    mae_list.append(predict_mae)
    rmse_list.append(predict_rmse)
    DF.append(df)

print('结束')
print(r2_list,mae_list,rmse_list)
print(sum(r2_list) / 5,sum(mae_list)/5,sum(rmse_list)/5)

end = datetime.datetime.now()
print('运行时间{}秒'.format(end-start))