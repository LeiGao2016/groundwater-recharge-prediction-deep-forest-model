import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import datetime
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn import metrics

start = datetime.datetime.now()

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

# feature_index = [0,13,12,5,11,9,4,1,7,6]
# new_x = np.zeros([x.shape[0],len(feature_index)])
# for i in range(len(feature_index)):
#     new_x[:,i] = x[:,feature_index[i]]
# x,y = new_x,y
print(x.shape)
# 直接进行预测
train_x, left_x, train_y, left_y = train_test_split(x, y, test_size=0.2, random_state=55)
val_x,test_x,val_y,test_y = train_test_split(left_x,left_y,test_size=0.5,random_state=25)
Lati_train,Lati_left,Longi_train,Longi_left = train_test_split(Lati,Longi,test_size=0.2,random_state=55)
Lati_val,Lati_test,Longi_val,Longi_test = train_test_split(Lati_left,Longi_left,test_size=0.5,random_state=25)


def ADA_CA(max_depth,n_estimators,learning_rate,random_state):
    folds = KFold(n_splits=5,shuffle=True,random_state=0)
    oof = np.zeros(x.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(x, y)):
        print("fold n°{}".format(fold_))
        x_train,y_train = x[trn_idx],y[trn_idx]
        x_val,y_val = x[val_idx],y[val_idx]

        clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=int(max_depth)),n_estimators=int(n_estimators), random_state=int(random_state),learning_rate=learning_rate)
        clf.fit(x_train,y_train)
        oof[val_idx] = clf.predict(x_val)

    return metrics.r2_score(oof,y)

pbounds = { 'max_depth':(2,6),
                'learning_rate': (0.1, 1),
                'n_estimators': (200,300),
                'random_state': (1, 100)}

adaboost_bo = BayesianOptimization(
            f=ADA_CA,   # 目标函数
            pbounds=pbounds,  # 取值空间
            verbose=1,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
        random_state=67,)

adaboost_bo.maximize(init_points=4,   # 随机搜索的步数
                         n_iter=20,       # 执行贝叶斯优化迭代次数
                         acq='ei')

print(adaboost_bo.max)
res = adaboost_bo.max
params_max = res['params']

model_adaboost = AdaBoostRegressor(
                     DecisionTreeRegressor(max_depth=int(params_max['max_depth'])),n_estimators=int(params_max['n_estimators']),
                     random_state=int(params_max['random_state']),learning_rate=params_max['learning_rate'])

model_adaboost.fit(train_x,train_y)

predict_y = model_adaboost.predict(test_x)
predict_R_2 = r2_score(test_y,predict_y)
predict_rmse = np.sqrt(mean_squared_error(test_y,predict_y))
predict_mae = mean_absolute_error(test_y,predict_y)


train_pre_y = model_adaboost.predict(train_x)
train_pre_R_2 = r2_score(train_y,train_pre_y)
train_pre_rmse = np.sqrt(mean_squared_error(train_y,train_pre_y))
train_pre_mae = mean_absolute_error(train_y,train_pre_y)

val_pre_y = model_adaboost.predict(val_x)
val_pre_R_2 = r2_score(val_y,val_pre_y)
val_pre_rmse = np.sqrt(mean_squared_error(val_y,val_pre_y))
val_pre_mae = mean_absolute_error(val_y,val_pre_y)

print("Train:--------------------------------")
print('train_pre_R_2',train_pre_R_2)
print('train_pre_rmse',train_pre_rmse)
print('train_pre_mae',train_pre_mae)

print("Test:---------------------------------")
print("predict_R_2",predict_R_2)
print("predict_rmse",predict_rmse)
print("predict_mae",predict_mae)

print("Validation:------------------------------")
print('val_pre_R_2',val_pre_R_2)
print('val_pre_rmse',val_pre_rmse)
print('val_pre_mae',val_pre_mae)


end = datetime.datetime.now()
print('运行时间{}秒'.format(end-start))
