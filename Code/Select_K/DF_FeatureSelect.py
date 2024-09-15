import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family']='Microsoft YaHei' #显示中文标签
plt.style.use ('ggplot') # 设定绘图风格
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")
import datetime
start = datetime.datetime.now()

class FeatureSelection:
    def __init__(self,n_estimators):
        self.n_estimators = n_estimators
    def select(self,x,y):
        # 运行随机森林和完全随机森林
        RF = ensemble.RandomForestRegressor(n_estimators=150, oob_score=True, bootstrap=True,max_features="sqrt",random_state=25)

        Bag_Before = ensemble.BaggingRegressor(base_estimator=RF, n_estimators=self.n_estimators, oob_score=True, bootstrap=True,
                                               bootstrap_features=10)
        Bag_Before.fit(x, y)
        error1 = 1 - Bag_Before.oob_score_
        print("error1:", error1)
        Score_list = []
        for i in range(x.shape[1]):
            np.random.permutation(x[:, i])  # 打乱第n列
            Bag_After = ensemble.BaggingRegressor(base_estimator=RF, n_estimators=self.n_estimators, oob_score=True, bootstrap=True,
                                                  bootstrap_features=10)
            Bag_After.fit(x, y)
            error2 = 1 - Bag_After.oob_score_
            Score_list.append(np.abs(error1 - error2))
            print("error",i, ":", error2)

        # 对Score_list进行归一化
        norm_score_list = []
        s_min, s_max = min(Score_list), max(Score_list)
        for s in Score_list:
            s_n = (s - s_min) / (s_max - s_min)
            norm_score_list.append(s_n)
        print(norm_score_list)

        indices = np.argsort(norm_score_list)[::-1] #对得分进行排序

        new_feature_importance = []   #对得分从大到小排序了
        for f in range(x.shape[1]):
            new_feature_importance.append(norm_score_list[indices[f]])
        print("执行select",indices)
        return indices,new_feature_importance

    def draw_importance(self,x,y):
        indices,new_feature_importance = self.select(x,y)
        x_columns = ['rainfall_annual_avg', 'aridity_index', 'LAI', 'NDVI', 'veg_height', 'rainfall_winter_avg',
                     'sand_depth_avg', 'silt_depth_avg', 'regolith_depth', 'MrVBF', 'rainfall_summer_avg',
                     'clay_depth_avg', 'PET', 'surface_geol', 'weathering_intensity_index']
        new_feature_name = []

        for f in range(x.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, x_columns[indices[f]], new_feature_importance[f]))
            new_feature_name.append(x_columns[indices[f]])

        #绘制图
        fig,ax = plt.subplots(figsize=(16, 9))

        # creating the bar plot
        ax.barh(new_feature_name,new_feature_importance,height =0.5, color='dodgerblue')

        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
                ax.spines[s].set_visible(False)

        # Remove x, y Ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Add padding between axes and labels
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=5)

        # Add x, y gridlines
        ax.grid(visible=True, linestyle='-.',linewidth=0.2, alpha=0.2)

        # Show top values
        ax.invert_yaxis()

        # Add annotation to bars   给柱状图添加数字注释
        for i in ax.patches:
                plt.text(i.get_width() + 0.01, i.get_y() + 0.3,
                         str(round((i.get_width()), 2)),
                         fontsize=10, fontweight='bold',
                         color='black')

        # Add Plot Title
        ax.set_title('Feature importance',
                     loc='left', )

        # Show Plot
        plt.savefig("Importance图")
        plt.show()
        print("执行了draw",indices)
