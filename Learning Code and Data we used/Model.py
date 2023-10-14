import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

sw_df = pd.read_csv(r"C:\Users\User\Desktop\DSCOVR PROJECT\dscovr data\dscovr data\output_dscovr.csv", sep=",")

kp_df = pd.read_csv(r"C:\Users\User\Desktop\DSCOVR PROJECT\dscovr data\dscovr data\output_kp.csv", sep=",") 

sws_ = sw_df["speed"]
d_ = sw_df["density"]
t_ = sw_df["temp"]
kp = kp_df["Kp"]

sws = {}
d = {}
t = {}

for i in range (1, 25):
  sws_i = sws_[i::3] / 100
  sws[str(i)] = sws_i
  
for i in range (1, 25):
  d_i = d_[i::3] / 10
  d[str(i)] = d_i

for i in range (1, 25):
  t_i = t_[i::3] / 100000
  t[str(i)] = t_i

kp = kp[9:]

kp = kp.reset_index(drop=True)

x_l = []

for i in range(1, 25):
    x = pd.concat([sws[str(i)], d[str(i)], t[str(i)]], axis=1)
    x_l.append(x)

for i in range(0, 24):
   x_l[i] = x_l[i].reset_index(drop=True)

x_df = pd.concat([x_l[23], x_l[22], x_l[21], x_l[20], x_l[19], x_l[18], x_l[17], x_l[16], x_l[15], x_l[14], x_l[13], x_l[12], x_l[11], x_l[10], x_l[9], x_l[8], x_l[7], x_l[6], x_l[5], x_l[4], x_l[3], x_l[2], x_l[1], x_l[0]], axis = 1)

x_df = x_df[:6239]

model = tf.keras.Sequential([ keras.layers.Dense(units=128, input_shape=[72]), keras.layers.Dense(units=4)])

model.compile(optimizer='Adam', loss='mse')

x_train, x_test, y_train, y_test = train_test_split(x_df, kp, test_size=0.1)

model.fit(x_train, y_train, epochs=350, use_multiprocessing=True, workers=16)

print(model.evaluate(x_test, y_test, batch_size=2) , "2")
print(model.evaluate(x_test, y_test, batch_size=4) , "4")
print(model.evaluate(x_test, y_test, batch_size=1) , "1")
print(model.evaluate(x_test, y_test, batch_size=8) , "8")

model.save(r"C:\Users\User\Desktop\DSCOVR PROJECT\dscovr data\dscovr data\models_new\model 1")
print("Saved")