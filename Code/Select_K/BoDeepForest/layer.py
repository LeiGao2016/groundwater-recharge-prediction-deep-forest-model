from sklearn.ensemble import ExtraTreesRegressor  # 引入极端森林回归
from sklearn.ensemble import RandomForestRegressor  # 引入随机森林回归
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np

class Layer:  # 定义层类
    def __init__(self,num_forests):
        self.num_forests = num_forests  # 定义森林数
        self.model = []  # 最后产生的类向量

    def RF_forest_optimization(self,x,y):    # 每一层的森林优化
        def RandomForest_val(n_estimators,max_depth,min_samples_split,max_features):
            val = cross_val_score(
                RandomForestRegressor(n_estimators=int(n_estimators),
                                      min_samples_split=int(min_samples_split),
                                      max_depth=int(max_depth),
                                      max_features=min(max_features, 0.999),
                                      n_jobs=10),
                x, y, scoring='neg_mean_squared_error', cv=5
            ).mean()
            return val

        # 参数取值空间
        RF_pbounds = {'n_estimators': (80, 150),  # 表示取值范围为50至150
                        'min_samples_split': (2, 30),
                        'max_depth': (3, 8),
                      'max_features':(0.1,0.999)}

        # 对随机森林进行参数优化
        RF_bo = BayesianOptimization(
            f=RandomForest_val,  # 目标函数
            pbounds=RF_pbounds,  # 取值空间
            verbose=0,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
        )

        RF_bo.maximize(init_points=6,  # 随机搜索的步数
                       n_iter=30,  # 执行贝叶斯优化迭代次数
                       acq='poi')

        RF_res = RF_bo.max
        RF_params_max = RF_res['params']
        # print('RF_params_max',RF_params_max)
        return RF_params_max

    def EX_forest_optimization(self, x, y):  # 每一层的森林优化
        def ExtraForest_val(n_estimators, max_depth,min_samples_split,max_features):
            val = cross_val_score(
                ExtraTreesRegressor(n_estimators=int(n_estimators),
                                    min_samples_split=int(min_samples_split),
                                    max_depth=int(max_depth),
                                    max_features=min(max_features, 0.999),
                                    n_jobs=10),
                x, y, scoring='neg_mean_squared_error', cv=5
            ).mean()
            return val

        EX_pbounds = {'n_estimators': (80, 150),  # 表示取值范围为50至150
                      'min_samples_split': (2, 30),
                      'max_depth': (3, 8),
                      'max_features':(0.1,0.999)}

        # 对完全随机森林进行参数优化
        EX_bo = BayesianOptimization(
            f=ExtraForest_val,  # 目标函数
            pbounds=EX_pbounds,  # 取值空间
            verbose=0,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
)

        EX_bo.maximize(init_points=6,  # 随机搜索的步数
                       n_iter=30,  # 执行贝叶斯优化迭代次数
                       acq='ei')

        EX_res = EX_bo.max
        EX_params_max = EX_res['params']
        # print('EX_params_max',EX_params_max)
        return EX_params_max

    def train(self,train_data, train_label):  # 训练函数、
        # 级联森林
        for forest_index in range(self.num_forests):  # 对具体的layer内的森林进行构建，num_forests是森林数
            if forest_index % 2 == 0:  # 如果是第偶数个设为随机森林
                RF_params_max = self.RF_forest_optimization(train_data, train_label)

                RF_n_estimators = int(RF_params_max['n_estimators'])
                RF_max_depth = int(RF_params_max['max_depth'])
                RF_min_samples_split = int(RF_params_max['min_samples_split'])
                RF_max_features = RF_params_max['max_features']

                clf = RandomForestRegressor(n_estimators=RF_n_estimators, # 子树的个数,
                                            n_jobs = -1,                    # cpu并行树，-1表示和cpu的核数相同
                                            max_depth = RF_max_depth,     # 最大深度,
                                            criterion = 'mse',
                                            min_samples_split = RF_min_samples_split,
                                            max_features=RF_max_features)

                clf.fit(train_data, train_label)
            else:  # 如果是第奇数个就设为极端森林
                EX_params_max = self.EX_forest_optimization(train_data, train_label)

                EX_n_estimators =  int(EX_params_max['n_estimators'])
                EX_max_depth = int(EX_params_max['max_depth'])
                EX_min_samples_split = int(EX_params_max['min_samples_split'])
                EX_max_features = EX_params_max['max_features']

                clf = ExtraTreesRegressor(n_estimators=EX_n_estimators, # 子树的个数,
                                          n_jobs = -1,                    # cpu并行树，-1表示和cpu的核数相同
                                          max_depth = EX_max_depth,     # 最大深度,
                                          criterion = 'mse',
                                          min_samples_split = EX_min_samples_split,
                                          max_features=EX_max_features)
                clf.fit(train_data, train_label)
            self.model.append(clf)  # 组建layer层


    def predict(self, test_data):  # 定义预测函数，也是最后一层的功能
        predict_prob = np.zeros([test_data.shape[0],self.num_forests])   # 列数是测试数据的样本数
        for forest_index, clf in enumerate(self.model):
            predict_prob[:,forest_index] = clf.predict(test_data)  # 分别得到每一个森林的预测数据
        predict_avg = np.mean(predict_prob,axis=1)
        return [predict_avg, predict_prob]


# 每一层的森林不用五折寻找最优
class CascadeForest:  # 定义一层级联森林并优化
    def __init__(self,num_forests,layer_index,train_data,train_label):
        self.num_forests = num_forests
        self.layer_index = layer_index
        self.model = Layer(self.num_forests)   # 初始化模型并训练
        self.model.train(train_data,train_label)

    def predict(self, test_data):  # 定义预测函数，用做下一层的训练数据
        test_prob,test_prob_concatenate = self.model.predict(test_data)
        return [test_prob, test_prob_concatenate]