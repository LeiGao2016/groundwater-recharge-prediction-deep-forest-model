import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import datetime
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

start = datetime.datetime.now()

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

# feature_index = [0,1,3,4,5,7,8,10,11,12]
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


def CAT_evaluate(learning_rate,l2_leaf_reg,depth,random_seed,subsample,metric_period):
    val = cross_val_score(
            CatBoostRegressor(
                iterations=100,
                learning_rate=learning_rate,
                    l2_leaf_reg=int(l2_leaf_reg),
                    depth= int(depth),
                    eval_metric='RMSE',
                    random_seed=int(random_seed),
                    bagging_temperature=1,
                    subsample=subsample,
                    od_type='Iter',
                    metric_period=int(metric_period),
                    od_wait=300),
                    x, y, scoring='neg_mean_squared_error', cv=5
        ).mean()
    return val

pbounds = {'learning_rate': (0.01, 0.999),
                   'l2_leaf_reg':(1,10),
                    'depth': (1, 10),
                    'random_seed': (1, 100),
                    'subsample': (0.01, 0.999),
                    'metric_period':(1,10)}

CAT_bo = BayesianOptimization(
            f=CAT_evaluate,   # 目标函数
            pbounds=pbounds,  # 取值空间
            verbose=1,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
            random_state=12,)

CAT_bo.maximize(init_points=4,   # 随机搜索的步数
                        n_iter=30,       # 执行贝叶斯优化迭代次数
                        acq='ei')

print(CAT_bo.max)
res = CAT_bo.max
params_max = res['params']

cat_model = CatBoostRegressor(iterations=100,
                                learning_rate=params_max['learning_rate'],
                                l2_leaf_reg= int(params_max['l2_leaf_reg']),
                                depth=int(params_max['depth']),
                                eval_metric='RMSE',
                                random_seed=int(params_max['random_seed']),
                                bagging_temperature=1,
                                subsample=params_max['subsample'],
                                od_type='Iter',
                                metric_period=int(params_max['metric_period']),
                                od_wait=300)

cat_model.fit(train_x,train_y)

predict_y = cat_model.predict(test_x)
predict_R_2 = r2_score(test_y,predict_y)
predict_rmse = np.sqrt(mean_squared_error(test_y,predict_y))
predict_mae = mean_absolute_error(test_y,predict_y)


train_pre_y = cat_model.predict(train_x)
train_pre_R_2 = r2_score(train_y,train_pre_y)
train_pre_rmse = np.sqrt(mean_squared_error(train_y,train_pre_y))
train_pre_mae = mean_absolute_error(train_y,train_pre_y)

val_pre_y = cat_model.predict(val_x)
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

