from BODF.Select_K.DeepForest.layer import *
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

# 定义gcforest模型
class gcForest:
    def __init__(self, num_forests,train_data,train_label,val_data,val_label):
        self.num_forests = num_forests  # 森林数量
        self.model = []
        self.new_model = []
        self.train_data = train_data    # 训练集
        self.train_label = train_label
        self.val_data = val_data        # 验证集
        self.val_label = val_label

    def train_and_predict(self,num_of_layer):
        layer_index = 1

        # 获得layer的train val 和 test数据
        layer = CascadeForest(self.num_forests,layer_index,self.train_data,self.train_label)  #先进行训练
        start_train_prob, start_train_stack = layer.predict(self.train_data)  # 获得第一层的输入
        start_val_prob,start_val_stack = layer.predict(self.val_data)

        start_val_score = 0
        self.model.append(layer)
        self.new_model.append(layer)
        # 这里加入k折交叉验证
        while layer_index<=num_of_layer:
            # print("layer " + str(layer_index))

            layer_train = np.concatenate([start_train_stack,self.train_data],axis = 1)
            layer_1 = CascadeForest(self.num_forests,layer_index,layer_train,self.train_label)
            first_train_prob, first_train_stack = layer_1.predict(layer_train)  #获得第一层的输入

            layer_val = np.concatenate([start_val_stack,self.val_data],axis = 1)
            first_val_prob,first_val_stack = layer_1.predict(layer_val)
            val_score = r2_score(self.val_label,first_val_prob)
            # print(val_score)

            start_train_stack = first_train_stack
            start_val_stack = first_val_stack
            # print("shape:",layer_train.shape,layer_val.shape)
            self.model.append(layer_1)
            self.new_model.append(layer_1)
            layer_index+=1

            # if val_score > start_val_score:
            #     start_val_score = val_score
            #     layer_index += 1
            #     self.new_model.append(layer_1)
            # else:
            #     print('layer_index:',layer_index)
            #     return self.model


    def predict_data(self,test_val):
        test = test_val
        test_result = np.zeros([test_val.shape[0]])
        for layer in self.model:
            temp_prob, temp_prob_concatenate = \
                layer.predict(test)    # 第一小曾预测出来的
            test = np.concatenate([temp_prob_concatenate,test_val], axis=1)
            # 将输出特征和原特征合并
            test_result = temp_prob
        return test_result

    def newpredict_data(self,test_val):
        test = test_val
        test_result = np.zeros([test_val.shape[0]])
        l=0
        for layer in self.new_model:
            l+=1
            temp_prob, temp_prob_concatenate = \
                layer.predict(test)    # 第一小曾预测出来的
            test = np.concatenate([temp_prob_concatenate,test_val], axis=1)
            # 将输出特征和原特征合并
            test_result = temp_prob
        print("new model layer",l)
        return test_result