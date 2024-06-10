import tensorflow as tf

#includeTENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
keras_model = tf.keras.models.load_model("ECG_net_tf.h5")
# 实例化转换器  
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)


# 设置支持的ops集，包括TFLITE_BUILTINS和SELECT_TF_OPS  
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]  
  
# 禁用降低张量列表操作的实验性标志  
# converter._experimental_lower_tensor_list_ops = False  
tflite_model = converter.convert()

open("./sden1.tflite","wb").write(tflite_model)

# conda -n tf2 python=3.7
# conda install cudatoolkit=11.3.1
# pip install tensorflow-gpu==2.10.0 -i  https://pypi.mirrors.ustc.edu.cn/simple

