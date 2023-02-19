import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

c = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
f = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1), metrics=['mean_squared_error'])
              
    
hist = model.fit(c, f, epochs=500, verbose=False)

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(hist.history['loss'])
plt.show()

c_pred =float(input("Digite a temperatura em Celsius (C°)\n"))


res = model.predict([c_pred])
res_output = "".join([str(resnum) for resnum in res])
res_output += " F"

print(res_output)

