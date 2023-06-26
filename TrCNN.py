import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers,models
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from math import sqrt
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings("ignore")

# ================== config begin ==================
train_input1 = "RT<=100.csv"

test_input = "RT>100h, 35 points.csv"
# ================== config end ==================

df_train = pd.read_csv(train_input1)
df_train = df_train.iloc[:,1:]

df_test = pd.read_csv(test_input)
df_test = df_test.iloc[:,1:]

df = pd.concat([df_train,df_test],axis=0)
train_size = df_train.shape[0]

scaler_x = MinMaxScaler()

X_all = df.drop("Ti_lg_RT", axis=1)
X_all_scaler = scaler_x.fit_transform(X_all)

x_data =[]
for i in range(len(X_all_scaler)):
    a = X_all_scaler[i,:]
    a = np.pad(a,(0,2),'constant',constant_values=(0,0))
    a = a.reshape(6,6,1)
    x_data.append(a)
x_data =np.array(x_data)

x_train = x_data[0:train_size]
x_test = x_data[train_size:]

y_all = df["Ti_lg_RT"]

y_train = y_all[0:train_size]
y_test = y_all[train_size:]

cnn_base = load_model('cnn.h5')
cnn_base.pop()
cnn_base.trainable = False
print(cnn_base.layers)


#
model = models.Sequential()
model.add(cnn_base)


model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))


model.summary()

#
adam = optimizers.Adam(lr=0.005)
model.compile(optimizer=adam, loss='mse')
checkpoint = ModelCheckpoint('cnn2.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(x_train, y_train, epochs=1500, batch_size=50,callbacks=[checkpoint],validation_data=[x_test,y_test])


m = load_model('cnn2.h5')
pre = m.predict(x_test).flatten()
pre2 = m.predict(x_train).flatten()
fig, ax = plt.subplots(figsize=(6,6))
plt.title('Trans_CNN_RT>100h, 35 points',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
ax.plot(pre2, y_train,"*",color='teal', label='Train Remainder');
ax.plot(pre, y_test,"*", color='navy', label='Test 35 points');
plt.plot([-3, 4], [-3, 4], '--')
plt.xlim((-3, 4))
plt.ylim((-3, 4))
plt.yticks(fontproperties='Times New Roman', size=12)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.ylabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
plt.xlabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
plt.legend(loc="upper left")
plt.savefig('Trans_CNN>100h, 35 points.png',dpi=500, bbox_inches='tight')
plt.show()
print(pre)
print(y_test)

mse = np.sum((y_test - pre) ** 2) / len(y_test)
rmse = sqrt(mse)
mae = np.sum(np.absolute(y_test - pre)) / len(y_test)
r2 = 1-mse/ np.var(y_test)#均方误差/方差
print(" mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2)

outpre = pd.DataFrame(pre)
outytest =pd.DataFrame(y_test)

# outpre.to_csv('pre_Ti_based.csv')
# outytest.to_csv('y_test_Ti_based.csv')

