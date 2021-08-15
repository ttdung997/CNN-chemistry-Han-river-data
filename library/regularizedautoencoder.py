from keras.models import Model, model_from_json
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import Dropout

class Antirectifier(layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Antirectifier, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.output_dim = output_dim
        self.a1 = self.add_weight(
            shape=(1, output_dim),
            initializer=self.initializer,
            name="a1",
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(1 , output_dim),
            initializer=self.initializer,
            name="b1",
            trainable=True,
        )
        self.a2 = self.add_weight(
            shape=(1, output_dim),
            initializer=self.initializer,
            name="a2",
            trainable=True,
        )
        self.b2 = self.add_weight(
            shape=(1 , output_dim),
            initializer=self.initializer,
            name="b2",
            trainable=True,
        )
        self.a3 = self.add_weight(
            shape=(1, output_dim),
            initializer=self.initializer,
            name="a3",
            trainable=True,
        )
        self.b3 = self.add_weight(
            shape=(1 , output_dim),
            initializer=self.initializer,
            name="b3",
            trainable=True,
        )
        self.a1 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)
        self.a2 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)
        self.a3 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        self.b1 = K.repeat_elements(K.expand_dims(10.0, axis = -1), self.output_dim, -1)
        self.b2 = K.repeat_elements(K.expand_dims(10.0, axis = -1), self.output_dim, -1)
        self.b3 = K.repeat_elements(K.expand_dims(10.0, axis = -1), self.output_dim, -1)
    
    # @tf.function
    def call(self, inputs):
        # inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        x = inputs
        # x1 = 1/(1+tf.math.exp(-self.a1*(x+self.b1)))
        # x2 = tf.math.exp(-tf.math.pow(x-self.b2,2)/(2*tf.math.pow(self.a2,2)))
        # x3 = 1/(1-tf.math.exp(self.a3*(x+self.b3)))
        # print(x)
        # print(self.a1)
        
        low = [0.0 for x in range(0,10)]
        low = [low for x in range(0,10)]
        low = tf.convert_to_tensor(low,dtype = "float")
        # print(x1)
        low = tf.reshape(low,(1,10,10))

        x1 = (x- self.a1)/(self.b1 - self.a1)
        x2 = (x- self.b1)/(self.a1 - self.b1)
        x3 = (x- self.b3)/(2*self.a3 - self.b3)
        # return x1

        high = [10.0 for x in range(0,10)]
        high = [high for x in range(0,10)]
        high = tf.convert_to_tensor(high,dtype = "float")
        # print(x1)
        high = tf.reshape(high,(1,10,10))

        
        return tf.concat([x1,x2,x3],-1)

        if tf.math.less_equal(x,self.a1):
            x1 = low
        elif tf.math.greater(x,self.a1)  and tf.math.greater(self.b1,x):
            x1 = (x- self.a1)/(self.b1 - self.a1)
        else:
            x1 = high
          
        if tf.math.less_equal(x,self.a2):
            x2 = low
        elif tf.math.greater(x,self.a2)  and tf.math.greater(self.b2,x):
            x2 = (x- self.a2)/(self.b2 - self.a2)
        else:
            x2 = high

        if tf.math.less_equal(x,self.a3) or tf.math.less_equal(2*self.a3,x) :
            x3 = low
        elif tf.math.greater(x,self.a3)  and tf.math.greater(self.b3,x):
            x3 = (x- self.a3)/(self.b3 - self.a3)
        elif tf.math.greater(x,self.b3)  and tf.math.greater(2*self.a3,x):
            x3 = (x- self.b3)/(2*self.a3 - self.b3)
        elif x == 2*self.a3:
            x3 = high
        else:
            x3 = low

        # x1 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        # x2 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        # x3 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        # x1 = (x- self.a1)/(self.b1 - self.a1)

        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        # print("WTD")
        # print(x)
        # print(x[0][0][0])
        # quit()
        # print(x1)
        # print(tf.concat([x1,x2,x3],-1))
        # print(tf.concat([x1,x2,x3],-1).shape)
        # quit()
        # xc = inputs*aligned_a
        # print(tf.concat([x1,x2,x3],-1).shape)
        return tf.concat([x1,x2,x3],-1)
        # return x1



class RegularizedAutoencoder(object):
    model_name = 'RegularizedAutoencoder'

    def __init__(self):
        self.model = None
        self.input_dim = None
        self.threshold = None
        self.config = None

    def load_model(self, model_dir_path):
        config_file_path = RegularizedAutoencoder.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.threshold = self.config['threshold']

        architecture_file_path = RegularizedAutoencoder.get_architecture_file_path(model_dir_path)
        self.model = model_from_json(open(architecture_file_path, 'r').read())
        weight_file_path = RegularizedAutoencoder.get_weight_file_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_model(self, input_dim, encoding_dim):

        # input_layer = Input(shape=(input_dim[0],input_dim[1],))

        input_layer = Antirectifier(shape=(input_dim[0],input_dim[1],))
        
        encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(10e-6))(input_layer)
        decoder = Dense(input_dim[1], activation='relu')(encoder)

        model = Model(inputs=input_layer, outputs=decoder)
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, RegularizedAutoencoder.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, RegularizedAutoencoder.model_name + '-weights.h5')

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, RegularizedAutoencoder.model_name + '-config.npy')

    def fit(self, input_data,output_data,input_dim, encoding_dim,
     model_dir_path, epochs=None, batch_size=None, test_size=None,
      random_state=None,estimated_negative_sample_ratio=None):
        start_time = time.time()
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if epochs is None:
            epochs = 10
        if batch_size is None:
            batch_size = 32
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        weight_file_path = RegularizedAutoencoder.get_weight_file_path(model_dir_path)
        architecture_file_path = RegularizedAutoencoder.get_architecture_file_path(model_dir_path)

        # X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)
        checkpointer = ModelCheckpoint(filepath=weight_file_path,
                                       verbose=0,
                                       save_best_only=True)
        self.input_dim = input_dim
        self.model = self.create_model(input_dim,encoding_dim)
        open(architecture_file_path, 'w').write(self.model.to_json())
        best_acc = 9999999.0
        best_iter = -1
        for ep in range(epochs):
            history = self.model.fit(input_data, output_data,
                                 epochs=1,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 verbose=1,
                                 callbacks=[checkpointer]).history

            output_predict = self.predict(input_data)
          
            acc = np.sqrt(((output_predict - output_data) ** 2).mean())
            print(acc)
            if acc < best_acc:
                best_iter = ep
                best_acc = acc
                best_model =  self.model
                # if acc > 0.93:
                #     break
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2:
                    break
        self.model = best_model
        self.model.save_weights(weight_file_path)

        print("best MSE: "+str(best_acc))
        print("--- %s seconds ---" % (time.time() - start_time))
        return history

    def predict(self, data):
        target_data = self.model.predict(x=data)
        # dist = np.linalg.norm(data - target_data, axis=-1)
        return target_data

    