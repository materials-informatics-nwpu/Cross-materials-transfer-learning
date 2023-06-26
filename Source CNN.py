import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers,models
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from math import sqrt

data = pd.read_csv('Superalloys.csv')
data = data.drop(['class'],axis=1)
# print(data)

x = data.iloc[:,:34]
y = data.iloc[:,34]

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_data =[]
for i in range(len(x)):
    a = x[i,:]
    print(a)
    a = np.pad(a,(0,2),'constant',constant_values=(0,0))
    print(a)
    a = a.reshape(6,6,1)
    x_data.append(a)
x_data =np.array(x_data)
print(x_data[0])


x_train, x_test, y_train, y_test_source = train_test_split(x_data, y,
                                                    train_size=0.8)


model = models.Sequential()
model.add(layers.Conv2D(8, (2, 2), activation='relu',padding='same', input_shape=(6, 6, 1)))
model.add(layers.Conv2D(16, (2, 2), padding='same',activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))
model.summary()


adam = optimizers.Adam(lr=0.005)
model.compile(optimizer=adam, loss='mse')
checkpoint = ModelCheckpoint('cnn.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(x_train, y_train, epochs=1500, batch_size=50,callbacks=[checkpoint],validation_data=[x_test,y_test_source])


m = load_model('cnn.h5')
pre_source = m.predict(x_test).flatten()
pre2_source = m.predict(x_train).flatten()
fig, ax = plt.subplots(figsize=(6,6))
plt.title('Superalloys_CNN',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
ax.plot(pre2_source, y_train,"D",color='aquamarine', label='Train');
ax.plot(pre_source, y_test_source,"D",color='pink',label='Test');
plt.plot([0, 6], [0, 6], '--',color='grey')
plt.xlim((0, 6))
plt.ylim((0, 6))
plt.yticks(fontproperties='Times New Roman', size=12)#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=12)
plt.ylabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
plt.xlabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
plt.legend(loc="upper left")
plt.savefig('Superalloys_CNN_lgRT.png',dpi=500, bbox_inches='tight')
plt.show()
print(pre_source)
print(y_test_source)



mse = np.sum((y_test_source - pre_source) ** 2) / len(y_test_source)
rmse = sqrt(mse)
mae = np.sum(np.absolute(y_test_source - pre_source)) / len(y_test_source)
r2 = 1-mse/ np.var(y_test_source)#均方误差/方差
print(" mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2)

outpre = pd.DataFrame(pre_source)
outytest =pd.DataFrame(y_test_source)

# outpre.to_csv('pre_superalloy.csv')
# outytest.to_csv('y_test_superalloy.csv')





