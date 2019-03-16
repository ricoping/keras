import numpy as np
from keras import models
from keras import layers
from neural import Neural
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras.utils.np_utils import to_categorical

class deepHouse(Neural):
	def data_preparing(self):
		(train_data, train_targets),(test_data, test_targets) = boston_housing.load_data()

		train_data, test_data = self.normalize(train_data, test_data)

		return (train_data, train_targets), (test_data, test_targets)

	def normalize(self, train_data, test_data):
		mean = train_data.mean(axis=0)
		std = train_data.std(axis=0)

		train_data -= mean; train_data /=std
		test_data -= mean; test_data /= std

		return train_data, test_data

	def create_models(self):
		model = models.Sequential()
		model.add(layers.Dense(64, activation='relu', input_shape=(self.x_train.shape[1],)))
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dense(1))
		model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

		return model

	def predict(self):
		pass

	def k_split_validation(self):
		k = 4
		num_val_samples = len(self.x_train) // k
		num_epochs = 100
		all_scores = []
		for i in range(k):
			print('processing fold #', i)

			val_data = self.x_train[i*num_val_samples:(i + 1)*num_val_samples]
			val_targets = self.train_labels[i*num_val_samples:(i + 1)*num_val_samples]

			partial_train_data = np.concatenate(
				[self.x_train[:i * num_val_samples],
				self.x_train[(i + 1)*num_val_samples:]], axis=0)

			partial_train_targets = np.concatenate(
				[self.train_labels[:i * num_val_samples],
				self.train_labels[(i + 1)*num_val_samples:]],axis=0)

			model = self.create_models()

			model.fit(partial_train_data, partial_train_targets,
						epochs=num_epochs, batch_size=1, verbose=0)
			try:
				val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
				all_scores.append(val_mae)
			except:
				continue
		print(all_scores)

if __name__ == "__main__":
	nn = deepHouse()
