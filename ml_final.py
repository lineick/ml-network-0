from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD
import argparse
import sys

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
    model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=1)

    return model

def eval_final_model(model):
    print("Loading dataset...")
        # load dataset
    trainX, trainY, testX, testY = load_dataset()
    print("Preparing pixel data...")
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    print("Loading model...")
    # summarize model.
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('> %.3f' % (acc * 100.0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and evaluate the final CNN model on the MNIST dataset.')
    parser.add_argument("--output", type=str, help="Path to save the final model to.")
    parser.add_argument("--eval", type=str, help="path of the final model to evaluate.")
    args = parser.parse_args()
    model = None

    if args.output:
        model = run_harness_for_final_model()
        print("\n----------------------\nSaving model to {}".format(args.output))
        model.save(args.output)
    else:
        print("[INFO]: no output path specified. Model will not be generated.")
    if args.eval and args.output:
        if args.eval == args.output:
            eval_final_model(model)
        else:
            print("[WARNING]: output and eval paths are different. Evaluating model from eval path...")
            model = load_model(args.eval)
            eval_final_model(model)
    elif args.eval:
        print("[INFO]: no output path specified. Evaluating model from eval path...")
        model = load_model(args.eval)
        eval_final_model(model)
    else:
        print("[INFO]: no eval path specified. Model will not be evaluated.")

    sys.exit(0)
