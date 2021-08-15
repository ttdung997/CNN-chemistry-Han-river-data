import csv
import numpy as np
import pandas as pd
import math
import os
from time import gmtime, strftime
import tensorflow as tf
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score,accuracy_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout


def report_evaluation_metrics(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    acc = accuracy_score(y_true,y_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Precision: {0:0.4f}'.format(precision))
    print('Recall: {0:0.4f}'.format(recall))
    print('F1: {0:0.4f}'.format(f1))
    print('acc: {0:0.4f}'.format(acc))

def get_data(dataset): 
	data= []
	with open(dataset, "r") as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row) 
		data =  np.array(data)
	return data

def Build_cnn_regression_model(input_shape, output_size, neurons, activ_func="linear",
    dropout=0.25, loss="mae", optimizer="adam"):
	model = Sequential()
	model.add(Conv1D(64, 5, activation='relu', input_shape = input_shape))

	model.add(MaxPooling1D(pool_size =2))

	model.add(Flatten())

	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.1))

	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))

	model.add(Dense(5, activation='linear'))


	model.compile(loss=loss, # one may use 'mean_absolute_error' as  mean_squared_error
	                  optimizer=optimizer,
	                  metrics=[tf.keras.metrics.RootMeanSquaredError()]# you can add several if needed
	                 )

	model.summary()

	return model

def main():
	folder_name = "data/EEM"
	col1 = 0
	col2 = 0
	files =[file for file in os.listdir(folder_name)]
	input_data = []
	for file in files:
		excel =  pd.ExcelFile("data/EEM/" + file)
		sheets =  excel.sheet_names

		for sheet in sheets:
			if sheet != "18b":
				row = excel.parse(sheet_name = sheet).values
				input_data.append(row)


	output_df = pd.read_excel("data/river_data.xlsx",sheet_name = "Data(2018-2020)")
	output = output_df[["TN,","DTP","TN","TP","pH"]].values

	
	input_data = np.array(input_data)
	output = np.array(output)
	my_model = Build_cnn_regression_model((len(input_data[0]),len(input_data[0][0])), output_size=5, neurons = 100)
	my_model.fit(input_data, output, epochs=3, batch_size=1)

	res = my_model.predict(input_data)
	print(res)

if __name__ == '__main__':
	main()
