from sklearn.ensemble import ExtraTreesRegressor  # 引入极端森林回归
from sklearn.ensemble import RandomForestRegressor  # 引入随机森林回归
import numpy as np

# 最普通的深度森林
class Layer:  # 定义层类
    def __init__(self,num_forests):
        self.num_forests = num_forests  # 定义森林数
        self.model = []  # 最后产生的类向量

    def train(self,train_data, train_label):  # 训练函数、
        # 级联森林
        for forest_index in range(self.num_forests):  # 对具体的layer内的森林进行构建，num_forests是森林数
            if forest_index % 2 == 0:  # 如果是第偶数个设为随机森林
                clf = RandomForestRegressor()
                clf.fit(train_data, train_label)
            else:  # 如果是第奇数个就设为极端森
                clf = ExtraTreesRegressor()
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