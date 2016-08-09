keras对tensorflow和theano进行的封装，它的变量和操作实际上就是tensorflow或theano的变量和操作。获取变量的方式都在`tensorflow_backend.py`或`theano_backend.py`中.

# Keras变量
1. 首先要导入backend:`import keras.backend as K`
2. 获取变量（theano中的shared变量，tensorflow的Variable）:
`K.variable(value,dtpye=_FLOATX,name=None)`
3. 获取输入数据的占位符：
    `K.placeholder(shape=None,ndim=None,dtype=_FLOATX,name=None)`

# Keras常量
1. `zeros(shape,dtype=_FLOATX,name=None)`：0矩阵
2. `K.ones(shape,dtype=_FLOATX,name=None)`: 1矩阵

# Keras标量、张量运算
`tensorflow_backend.py`和`theano.backend.py`中也定义了各种运算，加减乘除、矩阵运算等。

1. `K.dot(x,y)`: 点乘
2. `K.sqrt(x)`: 平方根=T.clip(x,0.,np.inf)
3. `K.exp(x)`
4. `K.log(x)`