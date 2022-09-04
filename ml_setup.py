from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from numpy import mean, std

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	i = 0
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		print(f"The model{i} started training...")
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
		print(f"The model{i} has successfully trained")
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=1)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
		i += 1
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	print("Loading dataset...")
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	print("Preparing pixel data...")
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	print("Evaluating model...")
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	print("Plotting learning curves...")
	summarize_diagnostics(histories)
	# summarize estimated performance
	print("Summarizing estimated performance...")
	summarize_performance(scores)

def run_harness_for_final_model():
    # load dataset
    print("Loading dataset...")
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    print("Preparing pixel data...")
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # fit model
    print("Fitting model...")
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=1)
    # save model
    print("Saving model...")
    model.save('final_model.h5')

def eval_final_model(path):
    print("Loading dataset...")
        # load dataset
    trainX, trainY, testX, testY = load_dataset()
    print("Preparing pixel data...")
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    print("Loading model...")
    # load the model
    model = load_model(path)
    # summarize model.
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('> %.3f' % (acc * 100.0))

# entry point, run the test harness
# run_harness_for_final_model()
eval_final_model('final_model.h5')
