# Merge Layer
可以根据`mode`把一个tensors列表合并成一个tensor。

```
#定义
calss Merge(Layer):
   def __init__(self,layers=None,
                mode='sum',
				concat_axis=-1,
				dot_axes=-1,
				output_shape=None,
				output_mask=None,
				node_indices=None,
				tensor_indices=None,
				name=None)

#参数
layers: A list of Keras tensors或 a list of layer instances.
mode: string或lambda/function。如果是string,必须是以下中的一个：'sum','mul','concat','ave','cos','max'。
concat_axis: integer, 'concat'使用的轴线。
dot_axes:integer或tuple of integers，'dot'或'cos'使用的坐标轴
output_shape:a shape tuple或者a lambda/function 来计算'output_shape'
node_indices: 可选
output_mask: mask或lambda/function来计算output mask(only if merge mode is a lambda/function)				
```