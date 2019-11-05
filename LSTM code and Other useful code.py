#Creat a LSTM model
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = x_train[:, 5:], y_train[:, 4]
	X = X.reshape(x_train.shape[1],1，x_train.shape[2])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, x_train.shape[1],x_train.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics = [matthews_correlation, 'acc'])
	for i in range(nb_epoch):
		model.fit(X, y, epochs=100, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

#The code for shifting dataset to apply lag function
def lagdata(ourdata, lag=1):
	dsf = DataFrame(ourdata)
	columns = [dsf.shift(1)]
	columns.append(dsf)
	dsf = concat(columns, axis=1)
	dsf.fillna(0, inplace=True)
	return dsf

#Code for creating difference between each measurement point
def difference(dsf, interval=1):
	difference = list()
	for i in range(interval, len(dsf)):
		value = dsf[i] - dataset[i - interval]
		diff.append(value)
	return Series(difference)

#Code for standardize dataset
def scale(x_train, x_test):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(x_train)
	x_train = train.reshape(x_train.shape[0], x_train.shape[1])
	train_scaled = scaler.transform(x_train)
	x_test = test.reshape(x_test.shape[0], x_test.shape[1])
	test_scaled = scaler.transform(x_test)
	return scaler, train_scaled, test_scaled

