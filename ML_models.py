import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,BatchNormalization,Conv1D,Activation,GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import numpy as np

def metric_balanced_accuracy_score(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    import tensorflow as tf
    from tensorflow.keras import backend as K
    return tf.py_func(balanced_accuracy_score, (K.greater(y_true,0.5), K.greater(y_pred,0.5)), tf.double)
def plot_history(history, what,gen=False):
    plt.figure(figsize=(30,4))
    plt.subplot(121)
    plt.plot(np.array(history.history[what]), label=what)
    plt.plot(np.array(history.history['val_'+what]), label='val_'+what)
    plt.ylabel(what); plt.xlabel('epoch'); plt.legend(); plt.grid()
    axes = plt.gca()
    plt.show()
def NeuralNetwork(X_train,y_train,X_val,y_val,batch_size,ClassWeights):    
    inputs = Input(shape=(X_train.shape[1],))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    num_epochs = 20
    history1 = model.fit(x = X_train, y = y_train, validation_data=(X_val, y_val), 
              epochs=num_epochs, batch_size=batch_size, shuffle=True,verbose=0,class_weight = {0: ClassWeights[0],1: ClassWeights[1]})
    #change learning rate
    optimizer = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history2 = model.fit(x = X_train, y = y_train, validation_data=(X_val, y_val),initial_epoch=num_epochs,
              epochs=num_epochs+4, batch_size=batch_size, shuffle=True,verbose=0,class_weight = {0: ClassWeights[0],1: ClassWeights[1]})
    history2 = model.fit(x = X_train, y = y_train, validation_data=(X_val, y_val),initial_epoch=num_epochs+4,
              epochs=num_epochs+5, batch_size=batch_size, shuffle=True,verbose=1,class_weight = {0: ClassWeights[0],1: ClassWeights[1]})
    return model

def ConvNet_1D(X_train,y_train,X_val,y_val,batch_size,ClassWeights):
    wd = 1e-3
    inputs = Input(shape=[X_train.shape[1],1])
    x = BatchNormalization()(inputs)
    x = Conv1D(32,kernel_size=9,strides=4,padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=l2(wd),use_bias=False)(x)
    for i in range(10):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if i <4:
            x = Conv1D(16,kernel_size=9,strides=1,padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=l2(wd),use_bias=False)(x)
        elif i == 4:
             x = Conv1D(32,kernel_size=9,strides=3,padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=l2(wd),use_bias=False)(x)
        elif i < 8:
             x = Conv1D(32,kernel_size=9,strides=1,padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=l2(wd),use_bias=False)(x)
        elif i == 8:
             x = Conv1D(32,kernel_size=9,strides=3,padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=l2(wd),use_bias=False)(x)
        else:
             x = Conv1D(64,kernel_size=9,strides=1,padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=l2(wd),use_bias=False)(x)
    x = Conv1D(1,kernel_size=9,strides=1,padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=l2(wd),use_bias=False)(x)
    x = GlobalAveragePooling1D()(x)
    predictions = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    optimizer = SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    y_cat = y_train
    y_cat_val = y_val
    history1 = model.fit(x=np.expand_dims(X_train,-1), 
                         y=y_cat,
                         validation_data=(np.expand_dims(X_val,-1),y_cat_val), 
                         epochs=5, 
                         batch_size=batch_size, 
                         shuffle=True,
                         verbose=1,
                         class_weight = {0: ClassWeights[0],1: ClassWeights[1]})
    # Change learning rate
    optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history2 = model.fit(x=np.expand_dims(X_train,-1), 
                         y=y_cat,
                         validation_data=(np.expand_dims(X_val,-1),y_cat_val), 
                         #callbacks=[es, mc], 
                          epochs=10,
                         initial_epoch=5,
                         batch_size=batch_size, 
                         shuffle=True,
                         verbose=1,
                         class_weight = {0: ClassWeights[0],1: ClassWeights[1]})
    optimizer = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history3 = model.fit(x=np.expand_dims(X_train,-1), 
                         y=y_cat,
                         validation_data=(np.expand_dims(X_val,-1),y_cat_val), 
                         epochs=11,
                         initial_epoch=10,
                         batch_size=batch_size, 
                         shuffle=True,
                         verbose=1,
                         class_weight = {0: ClassWeights[0],1: ClassWeights[1]})
    return model