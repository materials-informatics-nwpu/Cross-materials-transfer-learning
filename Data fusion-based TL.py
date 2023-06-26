import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.svm import LinearSVR
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
import sys
import xgboost as xgb
from xgboost import plot_importance
from statistics import mean
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2  # import the two-stage algorithm
import shap
import warnings
warnings.filterwarnings("ignore")



# ================== config begin ==================
train_input1 = "Superalloys.csv"
train_input2 = "eliminate 38 points.csv"

test_input = "T<=600, S>=300, 38 points.csv"
# ================== config end ==================



# 读取数据集 - 训练
df_train1 = pd.read_csv(train_input1)
df_train2 = pd.read_csv(train_input2)

#df_train1.rename(columns={"lg_RT":"Ti_lg_RT", "test_temperature":"Ti_test_temperature"}, inplace=True)
df_train = pd.concat([df_train1, df_train2])

# print(df_train1.shape, df_train2.shape, df_train.shape)

df_train = df_train.iloc[:,1:]

# 读取数据集 - 测试
df_test = pd.read_csv(test_input)
df_test = df_test.iloc[:,1:]

df = pd.concat([df_train,df_test],axis=0)
train_size = df_train.shape[0]

# min_max_scale
scaler_x = MinMaxScaler()


# 分离特征变量和目标变量
X_all = df.drop("lg_RT", axis=1)
X_all_scaler = scaler_x.fit_transform(X_all)

X_train = X_all_scaler[0:train_size]
X_test = X_all_scaler[train_size:]

# 分离特征变量和目标变量
y_all = df["lg_RT"]


y_train = y_all[0:train_size]
y_test = y_all[train_size:]



 def create_pic(sort_name):
     fig, ax = plt.subplots(figsize=(6,6))
     ax.plot(y_train_pred1, y_train[0:df_train1.shape[0]],"D",color='aquamarine', label='Train1');
     ax.plot(y_train_pred2, y_train[df_train1.shape[0]:],">",color='teal', label='Train2');
     ax.plot(y_test_pred, y_test,"*", color='navy', label='Test');
     plt.plot([0, 6], [0, 6], '--',color='grey');
     plt.xlim((0,6))
     plt.ylim((0,6))
     ax.set_ylabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
     ax.set_xlabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
     ax.tick_params(labelsize=12)
     plt.yticks(fontproperties='Times New Roman', size=12)#设置大小及加粗
     plt.xticks(fontproperties='Times New Roman', size=12)
     ax.legend(loc='best',prop='Times New Roman',markerscale=1,fontsize=60)
     ax.set_title(sort_name + '_',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
     plt.savefig(sort_name + '_.png',dpi=500, bbox_inches='tight')


 
# model 1 随机森林

 mae_list = []
 mse_list = []
 r2_list = []
 mape_list = []

 # 循环
 for i in range(1):

           
     parameters = {'n_estimators':[10,20,30],
                   'max_depth':[15,16,17],
                   'min_samples_split':[4,5,6],
                   'min_samples_leaf':[3,4,5]}
            
     #parameters = {'n_estimators':[20], 'max_depth':[14], 'min_samples_split':[1], 'min_samples_leaf':[2]}
     rfr = RandomForestRegressor()
     GS = GridSearchCV(estimator=rfr, param_grid=parameters, cv=5)
     GS.fit(X_train, y_train)

     print("随机森林的最佳超参数是：", GS.best_params_)
     # 随机森林的最佳超参数是： {'max_depth': 14, 'min_samples_leaf': 2, 'min_samples_split': 1, 'n_estimators': 20}

     # 用最佳超参数训练随机森林模型
     rfr_best = RandomForestRegressor(**GS.best_params_)

     # 计算训练集和测试集内的预测结果
     rfr_best.fit(X_train, y_train)
     y_train_pred1 = rfr_best.predict(X_train[0:df_train1.shape[0]])
     y_train_pred2 = rfr_best.predict(X_train[df_train1.shape[0]:])
     y_test_pred = rfr_best.predict(X_test)
    
     explainer = shap.Explainer(rfr_best.predict, X_all)
     shap_values = explainer(X_all)
     shap.summary_plot(shap_values, X_all, max_display=10)

     r2 = r2_score(y_test, y_test_pred)
     mae = mean_absolute_error(y_test, y_test_pred)
     mse = mean_squared_error(y_test, y_test_pred)
     mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
     mae_list.append(mae)
     mse_list.append(mse)
     r2_list.append(r2)
     mape_list.append(mape)
    
     print("随机森林第{}次循环,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

 print("随机森林各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
 print("")

 create_pic("RF")




 # model 2 高斯过程回归

 mae_list = []
 mse_list = []
 r2_list = []
 mape_list = []

 # 循环
 for i in range(1):


     parameters = {'normalize_y':[False],
                   'kernel':[None, DotProduct(), DotProduct() + WhiteKernel(), WhiteKernel()],
                   'alpha': np.arange(0.001, 0.1, 0.001)}
                  
     gpr = GaussianProcessRegressor()
     GS = GridSearchCV(estimator=gpr, param_grid=parameters, cv=5)
     GS.fit(X_train, y_train)
     print("高斯过程回归的最佳超参数是：", GS.best_params_)
     print("高斯过程回归的最佳超分数是：", GS.best_score_)


     gpr_best = GaussianProcessRegressor(**GS.best_params_)

     # 计算训练集和测试集内的预测结果
     gpr_best.fit(X_train, y_train)
     y_train_pred1 = gpr_best.predict(X_train[0:df_train1.shape[0]])
     y_train_pred2 = gpr_best.predict(X_train[df_train1.shape[0]:])
     y_test_pred = gpr_best.predict(X_test)
    

     explainer = shap.Explainer(gpr_best.predict, X_all)
     shap_values = explainer(X_all)
     shap.summary_plot(shap_values, X_all, max_display=10)

     r2 = r2_score(y_test, y_test_pred)
     mae = mean_absolute_error(y_test, y_test_pred)
     mse = mean_squared_error(y_test, y_test_pred)
     mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
     mae_list.append(mae)
     mse_list.append(mse)
     r2_list.append(r2)
     mape_list.append(mape)
    

     print("高斯过程回归第{}次循环,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

 print("高斯过程回归各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
 print("")

 create_pic("GPR")



 # model 3 SVR

 mae_list = []
 mse_list = []
 r2_list = []
 mape_list = []

 # 循环
 for i in range(3):

     parameters = {'C':[2,3,4,5,6,7,8],
                   'epsilon': np.arange(0.01, 1, 0.01)
                   }
     svr = LinearSVR()
     GS = GridSearchCV(estimator=svr, param_grid=parameters,cv=10)
     GS.fit(X_train, y_train)

     print("SVR的最佳超参数是：", GS.best_params_)

     # 用最佳超参数训练SVR模型
     svr_best = LinearSVR(**GS.best_params_)

     # 计算训练集和测试集内的预测结果
     svr_best.fit(X_train, y_train)
     y_train_pred1 = svr_best.predict(X_train[0:df_train1.shape[0]])
     y_train_pred2 = svr_best.predict(X_train[df_train1.shape[0]:])
     y_test_pred = svr_best.predict(X_test)

     r2 = r2_score(y_test, y_test_pred)
     mae = mean_absolute_error(y_test, y_test_pred)
     mse = mean_squared_error(y_test, y_test_pred)
     mape = mean_absolute_percentage_error(y_test, y_test_pred)

     mae_list.append(mae)
     mse_list.append(mse)
     r2_list.append(r2)
     mape_list.append(mape)

     print("SVR第{}次循环,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

 print("SVR各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
 print("")

 create_pic("SVR")


# model 4 xgboost

mae_list = []
mse_list = []
r2_list = []
mape_list = []

# 循环
 for i in range(1):

    # parameters = {'n_estimators':[140, 160],
    #               'max_depth': [4,5,6],
    #               'min_child_weight': [3,4,5],
    #               'gamma': [0.001, 0.01, 0.1],
    #               'subsample': [0.8, 0.9, 1.0],
    #               'colsample_bytree': [0.7, 0.8, 0.9],
    #               'reg_lambda': [3,5,10],
    #               'reg_alpha': [0, 0.1],
    #               }
    
    parameters = {'n_estimators':[160,170],
                  'max_depth': [5,6,7],
                  'min_child_weight': [2,3,4],
                  'gamma': [0.001, 0.01, 0.1],
                  'subsample': [0.7,0.8,0.9],
                  'colsample_bytree': [0.7, 0.8, 0.9],
                  'reg_lambda': [2,3,5,8],
                  'reg_alpha': [0,0.1],
                  }
    xgbreg = xgb.XGBRegressor()
    GS = GridSearchCV(estimator=xgbreg, param_grid=parameters,cv=5)
    GS.fit(X_train, y_train)

    print("xgboost的最佳超参数是：", GS.best_params_)
    # 最佳超参数是： {'colsample_bytree': 0.8, 'gamma': 0.01, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 160, 'reg_alpha': 0, 'reg_lambda': 40, 'subsample': 0.9}

    # 用最佳超参数训练xgb模型
    xgb_best = xgb.XGBRegressor(**GS.best_params_)

    # 计算训练集和测试集内的预测结果
    xgb_best.fit(X_train, y_train)
    y_train_pred1 = xgb_best.predict(X_train[0:df_train1.shape[0]])
    y_train_pred2 = xgb_best.predict(X_train[df_train1.shape[0]:])
    y_test_pred = xgb_best.predict(X_test)

    
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    mape_list.append(mape)

    print("xgb第{}次循环,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

 print("xgb各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
 print("")

 create_pic("XGBoost")




 # model 5 adaboost

 mae_list = []
 mse_list = []
 r2_list = []
 mape_list = []

 # 循环
 for i in range(1):
     # 划分训练集和测试集
     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

     # adaboost
     # 网格搜索调参
     parameters = {'n_estimators':[20,30,40,50,60, 80, 100],
                  }
                  
     ada = AdaBoostRegressor()
     GS = GridSearchCV(estimator=ada, param_grid=parameters,cv=5)
     GS.fit(X_train, y_train)

     print("adaboost的最佳超参数是：", GS.best_params_)

     # 用最佳超参数训练ada模型
     ada_best = AdaBoostRegressor(**GS.best_params_)

     # 计算训练集和测试集内的预测结果
     ada_best.fit(X_train, y_train)
     y_train_pred1 = ada_best.predict(X_train[0:df_train1.shape[0]])
     y_train_pred2 = ada_best.predict(X_train[df_train1.shape[0]:])
     y_test_pred = ada_best.predict(X_test)

     r2 = r2_score(y_test, y_test_pred)
     mae = mean_absolute_error(y_test, y_test_pred)
     mse = mean_squared_error(y_test, y_test_pred)
     mape = mean_absolute_percentage_error(y_test, y_test_pred)

     mae_list.append(mae)
     mse_list.append(mse)
     r2_list.append(r2)
     mape_list.append(mape)

     print("ada第{}次循环,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

 print("ada各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
 print("")

 create_pic("AdaBoost")
