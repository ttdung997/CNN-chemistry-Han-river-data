import csv
import numpy as np
import pandas as pd
import math
import os
from time import gmtime, strftime
import tensorflow as tf
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv1D, MaxPooling1D, Flatten,Permute,RepeatVector
from tensorflow.keras.layers import Dropout

def highlight_area(img, region, factor, outline_color=None, outline_width=1):
    """ Highlight specified rectangular region of image by `factor` with an
        optional colored  boarder drawn around its edges and return the result.
    """
    img = img.copy()  # Avoid changing original image.
    img_crop = img.crop(region)

    brightner = ImageEnhance.Brightness(img_crop)
    img_crop = brightner.enhance(factor)

    img.paste(img_crop, region)

    # Optionally draw a colored outline around the edge of the rectangular region.
    if outline_color:
        draw = ImageDraw.Draw(img)  # Create a drawing context.
        left, upper, right, lower = region  # Get bounds.
        coords = [(left, upper), (right, upper), (right, lower), (left, lower),
                  (left, upper)]
        draw.line(coords, fill=outline_color, width=outline_width)

    return img


def report_evaluation_metrics(y_true, y_pred):
	average_precision = average_precision_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
	recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1)
	f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)
	acc = accuracy_score(y_true, y_pred)
	print('Average precision-recall score: {0:0.2f}'.format(average_precision))
	print('Precision: {0:0.4f}'.format(precision))
	print('Recall: {0:0.4f}'.format(recall))
	print('F1: {0:0.4f}'.format(f1))
	print('acc: {0:0.4f}'.format(acc))


def get_data(dataset):
	data = []
	with open(dataset, "r") as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row)
		data = np.array(data)
	return data

# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # values shape == (batch_size, max_len, hidden size)

    # we are doing this to broadcast addition along the time axis to calculate the score
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

def Build_cnn_regression_model(input_shape, output_size, neurons, activ_func="linear",
							   dropout=0.25, loss="mae", optimizer="adam"):
	model = Sequential()
	model.add(Conv1D(64, 5, activation='relu', input_shape=input_shape))



	model.add(MaxPooling1D(pool_size=2))

	# model.add(BahdanauAttention(100))
	# model.add(Flatten())

	model.add(Dense(512, activation='relu'))

	model.add(Dense(256, activation='relu'))


	#attention layer 
	model.add(Dense(1, activation='tanh'))
	
	model.add(Flatten())
	
	model.add(Activation('softmax'))
	
	model.add(RepeatVector(128))
	
	model.add(Permute([2, 1]))

	#end 
	model.add(Flatten())
	model.add(Dropout(0.1))
	model.add(Dense(128, activation='relu'))



	model.add(Dense(13, activation='linear'))

	model.compile(loss=loss,  # one may use 'mean_absolute_error' as  mean_squared_error
				  optimizer=optimizer,
				  metrics=[tf.keras.metrics.RootMeanSquaredError()]  # you can add several if needed
				  )
	model.summary()
	return model


def split_data(data: list, ratio=0.9):
	size = len(data)
	train_sample = int(size * ratio)
	train_dataset, test_dataset = data[: train_sample], data[train_sample:]
	return np.array(train_dataset), np.array(test_dataset)


def mse(label: np.ndarray, predict: np.ndarray):
	sum_square = (label - predict) ** 2
	return np.mean(sum_square)


def mpe(label: np.ndarray, predict: np.ndarray):
	sum_square = (label - predict) ** 2 / np.average(label)*100
	return np.mean(sum_square)

def _pprint(field, output_test, res):
	total_loss = 0
	for i in range(len(field)):
		loss = mse(output_test[:, i], res[:, i])
		total_loss += loss
		print(f"Field: {field[i]} Mse: {loss}")
	print(f"Total Loss MSE: {total_loss}")

	print("_____________________________")

	total_loss = 0
	for i in range(len(field)):
		loss = mpe(output_test[:, i], res[:, i])
		total_loss += loss
		print(f"Field: {field[i]} Mpe: {loss} %")

def main():
	folder_name = "data/EEM"
	col1 = 0
	col2 = 0
	files = [file for file in os.listdir(folder_name)]
	input_data = []
	for file in files:
		excel = pd.ExcelFile(os.path.join(folder_name, file))
		sheets = excel.sheet_names

		for sheet in sheets:
			if sheet != "18b":
				row = excel.parse(sheet_name=sheet).values
				input_data.append(row)

	full_col =['pH', 'DO', 'BOD5', 'CODMn', 'TN', 'TP', 'TOC', 'DOC',
	 'TN,', 'NH3-N', 'NO3-N', 'DTP', 'PO4-P']
	
	field = full_col
	# field = ["TN,", "DTP", "TN", "TP"]

	output_df = pd.read_excel("data/river_data.xlsx", sheet_name="Data(2018-2020)")
	print(output_df.columns)
	output = output_df[field].values

	train_dataset, test_dataset = split_data(input_data)
	input_data = np.array(input_data)
	output_train = np.array(output[:len(train_dataset)])
	output_test = np.array(output[len(train_dataset):len(input_data)])
	my_model = Build_cnn_regression_model((len(train_dataset[0]), len(train_dataset[0][0])), output_size=5, neurons=100)
	my_model.fit(train_dataset, output_train, epochs=5, batch_size=1)
	res = my_model.predict(test_dataset)
	print(output)
	print(output_test.shape)
	print(res.shape)
	_pprint(field, output_test, res)



if __name__ == '__main__':
	main()
