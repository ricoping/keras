from neural import Neural
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

class deepReuters(Neural):
	def data_preparing(self):
		(train_data, train_labels),(test_data, test_labels) = reuters.load_data(num_words=10000)

		x_train = self.vectorize_sequences(train_data)
		x_test = self.vectorize_sequences(test_data)

		one_hot_train_labels = to_categorical(train_labels)
		one_hot_test_labels = to_categorical(test_labels)

		return (x_train, one_hot_train_labels), (x_test, one_hot_test_labels)


	def create_models(self):
		model = models.Sequential()
		model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dense(46, activation='softmax'))

		model.compile(optimizer='rmsprop',
			loss='categorical_crossentropy', metrics=['accuracy'])

		return model

if __name__ == "__main__":
	nn = deepReuters(epochs=5, batch_size=512)