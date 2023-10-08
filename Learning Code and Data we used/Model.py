import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split


# Solar_Wind_data
sw_df = pd.read_csv(r"C:\Users\99899\Documents\dscovr data\output_dscovr.csv", sep=",")
# Kp_data
kp_df = pd.read_csv(r"C:\Users\99899\Documents\dscovr data\output_kp.csv", sep=",") 

#New Vars
bx = sw_df["bX"]
by = sw_df["bY"]
bz = sw_df["bZ"]
sws = sw_df["speed"]
d = sw_df["density"]
t = sw_df["temp"]
kp = kp_df["Kp"]

# Split_pre
bx_3 = bx[:-3:3]
bx_2 = bx[1:-2:3]
bx_1 = bx[2:-1:3]

by_3 = by[:-3:3]
by_2 = by[1:-2:3]
by_1 = by[2:-1:3]

bz_3 = bz[:-3:3]
bz_2 = bz[1:-2:3]
bz_1 = bz[2:-1:3]

sws_3 = sws[:-3:3]/100
sws_2 = sws[1:-2:3]/100
sws_1 = sws[2:-1:3]/100

d_3 = d[:-3:3]/10
d_2 = d[1:-2:3]/10
d_1 = d[2:-1:3]/10

t_3 = t[:-3:3]/100000
t_2 = t[1:-2:3]/100000
t_1 = t[2:-1:3]/100000


kp = kp[1:]

kp = kp.reset_index(drop=True)
# kp_df = pd.concat ([kp], axis=1)


#to DataFrame
x1 = pd.concat([bx_1, by_1, bz_1, sws_1, d_1, t_1], axis=1)
x2 = pd.concat([bx_2, by_2, bz_2, sws_2, d_2, t_2], axis=1)
x3 = pd.concat([bx_3, by_3, bz_3, sws_3, d_3, t_3], axis=1)

x1 = x1.reset_index(drop=True)
x2 = x2.reset_index(drop=True)
x3 = x3.reset_index(drop=True)

x_df = pd.concat ([x1, x2, x3], axis=1)

# Split
x_train, x_test, y_train, y_test = train_test_split(x_df, kp, test_size=0.2, random_state = 43)

model = tf.keras.Sequential([ keras.layers.Dense(units=128, input_shape=[18]), keras.layers.Dense(units=4) ])

model.compile(optimizer='adam', loss='mae')

model.fit(x_train, y_train, epochs=800, validation_split=0.2)

print(model.evaluate(x_test, y_test, batch_size=64))

model.save(r"C:\Users\99899\Documents\dscovr data\models\Model")
print("Saved")