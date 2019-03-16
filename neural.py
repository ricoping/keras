import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical


class Neural():
	def __init__(self, epochs=20, batch_size=512, verbose=1, validation_size=1000):
		self.epochs = epochs
		self.batch_size = batch_size
		self.verbose = verbose
		self.validation_size = validation_size

		(self.x_train, self.train_labels), (self.x_test, \
			self.test_labels) = self.data_preparing()
		self.data_description()

		self.model = self.create_models()
		print(self.model.summary())

		self.train()

		print("\nPREDICTION OF 200 DATA","-"*90, "\n")
		self.predict()

		self.test()

		self.k_split_validation()

		self.generate()

	def vectorize_sequences(self, sequences, dimension=10000):
		results = np.zeros((len(sequences), dimension))
		for i, sequence in enumerate(sequences):
			results[i, sequence] = 1.
		return results

	def data_description(self):
		print("DATA DESCRIPTION", "-"*90)
		print("x_train, x_test")
		print(self.x_train.shape, self.x_test.shape, "\n")
		print("train_labels, test_labels")
		print(self.train_labels.shape, self.test_labels.shape)	
		print("-"*100, "\n")

	def train(self):
		x_val = self.x_train[:self.validation_size]
		partial_x_train = self.x_train[self.validation_size:]

		y_val = self.train_labels[:self.validation_size]
		partial_y_train = self.train_labels[self.validation_size:]
		
		history = self.model.fit(
							partial_x_train,
							partial_y_train,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=(x_val, y_val),
							verbose=self.verbose)

		history_dict = history.history
		loss_values = history_dict['loss']
		val_loss_values = history_dict['val_loss']

		self.drow_figure(history_dict, loss_values, val_loss_values)


	def test(self):
		self.model.fit(self.x_test, self.test_labels, epochs=self.epochs, batch_size=self.batch_size)
		test_loss, test_acc = self.model.evaluate(self.x_test, self.test_labels)
		print("test_loss", test_loss)
		print("test_acc", test_acc)

	def drow_figure(self, history_dict, loss_values, val_loss_values):
		epochs = range(1, len(loss_values) + 1)

		plt.plot(epochs, loss_values, 'bo', label="Training Loss")
		plt.plot(epochs, val_loss_values, 'b', label="Validation Loss")
		plt.title("Training and validation loss")
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig("neural.jpg")

		plt.clf()

	def predict(self):
		predictions = self.model.predict(self.x_test)

		for i, pre in enumerate(predictions[:200]):
			p = np.argmax(pre)
			print("Prediction of index '%s' of x_test is %s. The answer is %s" % (str(i), str(p), np.argmax(self.test_labels[i])))

	def data_preparing(self):
		pass

	def create_models(self):
		pass

	def k_split_validation(self):
		pass

	def generate(self):
		pass

if __name__ == "__main__":
	nn = Neural()