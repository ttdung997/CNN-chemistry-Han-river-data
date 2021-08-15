from keras.models import Model, model_from_json
from keras.layers import Input, Dense
from keras.layers import TimeDistributed
 
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, GRU, Bidirectional, RepeatVector, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
import time

class ConvAutoencoder(object):
    model_name = 'ConvAutoencoder'

    def __init__(self):
        self.model = None
        self.input_dim = None
        self.threshold = None
        self.config = None

    def load_model(self, model_dir_path):
        config_file_path = ConvAutoencoder.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.threshold = self.config['threshold']

        architecture_file_path = ConvAutoencoder.get_architecture_file_path(model_dir_path)
        self.model = model_from_json(open(architecture_file_path, 'r').read())
        weight_file_path = ConvAutoencoder.get_weight_file_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_model(self, input_dim, encoding_dim):
        model = Sequential()
        model.add(GRU(100, activation='relu', input_shape=(24,6)))
        model.add(RepeatVector(24))
        model.add(GRU(100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(6)))
        model.compile(optimizer='adam', loss='mse')

        return model

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, ConvAutoencoder.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, ConvAutoencoder.model_name + '-weights.h5')

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, ConvAutoencoder.model_name + '-config.npy')

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

        weight_file_path = ConvAutoencoder.get_weight_file_path(model_dir_path)
        architecture_file_path = ConvAutoencoder.get_architecture_file_path(model_dir_path)

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

    