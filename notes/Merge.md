# Merge Layer
���Ը���`mode`��һ��tensors�б�ϲ���һ��tensor��

```
#����
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

#����
layers: A list of Keras tensors�� a list of layer instances.
mode: string��lambda/function�������string,�����������е�һ����'sum','mul','concat','ave','cos','max'��
concat_axis: integer, 'concat'ʹ�õ����ߡ�
dot_axes:integer��tuple of integers��'dot'��'cos'ʹ�õ�������
output_shape:a shape tuple����a lambda/function ������'output_shape'
node_indices: ��ѡ
output_mask: mask��lambda/function������output mask(only if merge mode is a lambda/function)				
```