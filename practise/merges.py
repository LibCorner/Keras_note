#encode
from keras.layers import Merge,Dense
from keras.models import Sequential,Model
import numpy as np

model1=Sequential()
model1.add(Dense(10,input_dim=10))

model2=Sequential()
model2.add(Dense(10,input_dim=10))

model=Sequential()
#tensor相加
model.add(Merge(layers=[model1,model2],mode="sum"))
model.compile(optimizer="sgd",loss="mse")




#dot/cos
model_dot=Sequential()
#model_dot.add(Merge(layers=[model1,model2],mode="dot",dot_axes=1))
model_dot.add(Merge(layers=[model1,model2],mode="cos",dot_axes=1))
model_dot.compile(optimizer="sgd",loss="mse")

#Test
data1=np.random.rand(2,10)
data2=np.random.rand(2,10)
pre1=model1.predict(data1)
pre2=model2.predict(data2)
pre=model_dot.predict([data1,data2])