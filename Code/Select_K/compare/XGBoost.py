import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import datetime
from bayes_opt import BayesianOptimization

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

# feature_index = [0,1,3,4,5,7,8,10,11,12]
# new_x = np.zeros([x.shape[0],len(feature_index)])
# for i in range(len(feature_index)):
#     new_x[:,i] = x[:,feature_index[i]]
# x,y = new_x,y

# 直接进行预测
train_x, left_x, train_y, left_y = train_test_split(x, y, test_size=0.2, random_state=55)
val_x,test_x,val_y,test_y = train_test_split(left_x,left_y,test_size=0.5,random_state=25)
Lati_train,Lati_left,Longi_train,Longi_left = train_test_split(Lati,Longi,test_size=0.2,random_state=55)
Lati_val,Lati_test,Longi_val,Longi_test = train_test_split(Lati_left,Longi_left,test_size=0.5,random_state=25)


def evaluate(x_train,y_train):
    dtrain = xgb.DMatrix(x_train, label=y_train)

    def _xgb_evaluate(max_depth, subsample,gamma,reg_alpha,seed,colsample_bytree,min_child_weight,learning_rate):
        params = {'eval_metric': 'rmse',
                        'max_depth': int(max_depth),
                        'subsample': subsample,
                        'eta': 0.3,
                        'gamma': gamma,
                        'colsample_bytree': colsample_bytree,
                        'min_child_weight': min_child_weight,
                        'max_delta_step': 0,
                        'lambda': 1,
                        'alpha': 0.1,
                        'reg_alpha': reg_alpha,
                        'reg_lambda': 4,
                        'seed': int(seed),
                        'nthread': 0,
                        'scale_pos_weight': 0,
                        'random_state': 0,
                        'learning_rate': learning_rate}
        cv_result = xgb.cv(params, dtrain, num_boost_round=122, nfold=5)  # nfold 分五组交叉验证

        return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

    xgb_bo = BayesianOptimization(_xgb_evaluate, {'max_depth': (3, 11),
                                                      'subsample':(0, 1),
                                                      'gamma': (0, 1),
                                                       'colsample_bytree': (0.1, 0.9),
                                                       'min_child_weight': (1,100),
                                                       'reg_alpha':(0, 1),
                                                       'seed':(1,100),
                                                       'learning_rate':(0,1)},random_state=2,verbose=1)

    xgb_bo.maximize(init_points=4, n_iter=100, acq='ucb')

    print(xgb_bo.max)
    res = xgb_bo.max
    params_max = res['params']
    return params_max

params_max = evaluate(x_train=train_x,y_train = train_y)   # 获得最优参数

# XGBoost predict
params = {'objective': 'reg:linear',
                'booster':'gbtree',
                'importance_type':'gain',
                'n_jobs':1,
                'n_estimators':200,
                'max_delta_step':0,
                'max_depth': int(params_max['max_depth']),
                'lambda': 1,
                'subsample': params_max['subsample'],
                'colsample_bytree': params_max['colsample_bytree'],
                'min_child_weight': params_max['min_child_weight'],
                'alpha': 0.1,
                'reg_alpha':params_max['reg_alpha'],
                'reg_lambda':4,
                'seed':int(params_max['seed']),
                'nthread':0,
                'gamma': params_max['gamma'],
                'scale_pos_weight' : 0,
                'random_state' :0,
                'learning_rate': params_max['learning_rate']}

dtrain = xgb.DMatrix(train_x, label=train_y)
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=122, evals=watchlist)

dtest = xgb.DMatrix(test_x)
predict_y = bst.predict(dtest)
predict_R_2 = r2_score(test_y,predict_y)
predict_rmse = np.sqrt(mean_squared_error(test_y,predict_y))
predict_mae = mean_absolute_error(test_y,predict_y)

dtrain = xgb.DMatrix(train_x)
train_pre_y = bst.predict(dtrain)
train_pre_R_2 = r2_score(train_y,train_pre_y)
train_pre_rmse = np.sqrt(mean_squared_error(train_y,train_pre_y))
train_pre_mae = mean_absolute_error(train_y,train_pre_y)

dval = xgb.DMatrix(val_x)
val_pre_y = bst.predict(dval)
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












# R_2 = []
# for i in range(100):
#     # 直接进行预测
#     print("这是第{}次测验".format(i))
#     from sklearn.model_selection import train_test_split
#     x_train, x_left, y_train, y_left = train_test_split(x, y, test_size=0.2, random_state=55)
#     x_val,x_test,y_val,y_test = train_test_split(x_left,y_left,test_size=0.5,random_state=78)
#
#
#     def evaluate(x_train,y_train):
#         dtrain = xgb.DMatrix(x_train, label=y_train)
#
#         def _xgb_evaluate(max_depth, subsample,gamma,reg_alpha,seed,colsample_bytree,min_child_weight,learning_rate):
#             params = {'eval_metric': 'rmse',
#                         'max_depth': int(max_depth),
#                         'subsample': subsample,
#                         'eta': 0.3,
#                         'gamma': gamma,
#                         'colsample_bytree': colsample_bytree,
#                         'min_child_weight': min_child_weight,
#                         'max_delta_step': 0,
#                         'lambda': 1,
#                         'alpha': 0.1,
#                         'reg_alpha': reg_alpha,
#                         'reg_lambda': 4,
#                         'seed': int(seed),
#                         'nthread': 0,
#                         'scale_pos_weight': 0,
#                         'random_state': 0,
#                         'learning_rate': learning_rate}
#             cv_result = xgb.cv(params, dtrain, num_boost_round=122, nfold=5)  # nfold 分五组交叉验证
#
#             return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
#
#         xgb_bo = BayesianOptimization(_xgb_evaluate, {'max_depth': (3, 11),
#                                                       'subsample':(0, 1),
#                                                       'gamma': (0, 1),
#                                                        'colsample_bytree': (0.1, 0.9),
#                                                        'min_child_weight': (1,100),
#                                                        'reg_alpha':(0, 1),
#                                                        'seed':(1,100),
#                                                        'learning_rate':(0,1)},random_state=2,verbose=0,)
#
#         xgb_bo.maximize(init_points=4, n_iter=100, acq='ucb')
#
#         print(xgb_bo.max)
#         res = xgb_bo.max
#         params_max = res['params']
#         return params_max
#
#     params_max = evaluate(x_train=x_train,y_train = y_train)   # 获得最优参数
#
#     # XGBoost predict
#     params = {'objective': 'reg:linear',
#                 'booster':'gbtree',
#                 'importance_type':'gain',
#                 'n_jobs':1,
#                 'n_estimators':200,
#                 'max_delta_step':0,
#                 'max_depth': int(params_max['max_depth']),
#                 'lambda': 1,
#                 'subsample': params_max['subsample'],
#                 'colsample_bytree': params_max['colsample_bytree'],
#                 'min_child_weight': params_max['min_child_weight'],
#                 'alpha': 0.1,
#                 'reg_alpha':params_max['reg_alpha'],
#                 'reg_lambda':4,
#                 'seed':int(params_max['seed']),
#                 'nthread':0,
#                 'gamma': params_max['gamma'],
#                 'scale_pos_weight' : 0,
#                 'random_state' :0,
#                 'learning_rate': params_max['learning_rate']}
#
#     dtrain = xgb.DMatrix(x_train, label=y_train)
#     dtest = xgb.DMatrix(x_test)
#     watchlist = [(dtrain, 'train')]
#
#     bst = xgb.train(params, dtrain, num_boost_round=122, evals=watchlist)
#     predict_y = bst.predict(dtest)
#     r_2 = r2_score(y_test,predict_y)
#     mse,mae = mean_squared_error(y_test,predict_y),mean_absolute_error(y_test,predict_y)
#
#
#     print("r_2:",r_2)
#     print('rmse:',np.sqrt(mse))
#     R_2.append(r_2)
#
# print(R_2.index(max(R_2)))


