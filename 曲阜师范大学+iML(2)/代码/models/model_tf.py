from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Dropout,Input, Conv2D, BatchNormalization, Activation, Concatenate, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Dense  
  
def dense_block(x, blocks, growth_rate):    
    for i in range(blocks):    
        cb = BatchNormalization()(x)    
        cb = Activation('relu')(cb)    
        cb1 = Conv2D(growth_rate * 4, kernel_size=(1, 1), padding='same')(cb)          # 瓶颈层减少计算量  
        cb1 = BatchNormalization()(cb1)  
        cb1 = Activation('relu')(cb1)  
        cb = Conv2D(growth_rate, kernel_size=(3, 1), padding='same')(cb1)  
        x = Concatenate()([x, cb])    
    return x    
  
def transition_block(x, reduction):    
    bn = BatchNormalization()(x)    
    act = Activation('relu')(bn)    
    conv = Conv2D(int(x.shape[-1] * reduction), kernel_size=(1, 1), padding='same')(act)  # 使用x.shape[-1]来获取当前通道数  
    pool = AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(conv)    
    return pool    
  
def DenseNet(input_shape, growth_rate=16, block_config=(4, 8, 12 )):  
    inputs = Input(shape=input_shape)  
      
    x = Conv2D(16, kernel_size=(2, 1), strides=(2, 1), padding='same')(inputs)  
    x = BatchNormalization()(x)  
    x = Activation('relu')(x)  
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)
    # X = Dropout(0.2)(x)
      
    num_filters = 16  
    for i, blocks in enumerate(block_config):  
        x = dense_block(x, blocks, growth_rate)  
        num_filters += blocks * growth_rate  
          
        if i != len(block_config) - 1:  
            x = transition_block(x, 0.5)  
            num_filters = int(num_filters * 0.5)  
       
    x = GlobalAveragePooling2D()(x)  

    outputs = Dense(2, activation='softmax')(x)  
      
    model = Model(inputs=inputs, outputs=outputs, name='DenseNet')  
    return model  
  
def AFNet(input_shape):  
    model = DenseNet(input_shape)  
    return model