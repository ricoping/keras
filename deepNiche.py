import random, sys, pprint
import numpy as np
import mecabing
from neural import Neural
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

class deepNiche(Neural):
	def data_preparing(self):
		maxlen = 60   ; step = 3
		sentences = []; next_chars = []
		f = open("terada.txt"); text = f.read().lower(); f.close()
		#text = mecabing.word_split(text)
		#text = text.split(" ")[:-3]
		print('Corpus length: ', len(text))


		for i in range(0, len(text) - maxlen, step):
			sentences.append(text[i: i + maxlen])
			next_chars.append(text[i + maxlen])

		print('Number of sequences: ', len(sentences))
		print(sentences[:10])
		print(next_chars[:10])

		chars = sorted(list(set(text)))
		print('Unique characters: ', len(chars))

		char_indices = dict((char, chars.index(char)) for char in chars)

		print('Vectorization...')

		x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
		y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
		for i, sentence in enumerate(sentences):
			for t, char in enumerate(sentence):
				x[i, t, char_indices[char]] = 1
			y[i, char_indices[next_chars[i]]] = 1


		self.x = x
		self.y = y
		self.chars = chars
		self.maxlen = maxlen
		self.text = text
		self.char_indices = char_indices

		return (np.array([]),np.array([])), (np.array([]),np.array([]))

	def create_models(self):
		model = models.Sequential()
		model.add(layers.LSTM(128, input_shape=(self.maxlen, len(self.chars))))
		model.add(layers.Dense(len(self.chars), activation='softmax'))

		optimizer = optimizers.RMSprop(lr=0.01)

		model.compile(optimizer=optimizer, loss='categorical_crossentropy')

		return model

	def sample(self, preds, temperature=1.0):
		preds = np.asarray(preds).astype('float64')
		preds = np.log(preds) / temperature
		exp_preds = np.exp(preds)
		preds = exp_preds / np.sum(exp_preds)
		probas = np.random.multinomial(1, preds, 1)

		return np.argmax(probas)

	def generate(self):
		x = self.x
		y = self.y
		maxlen = self.maxlen
		text = self.text
		chars = self.chars
		char_indices = self.char_indices
		for epoch in range(1, 60):
			print('epoch', epoch)

			self.model.fit(x, y, batch_size = 128, epochs = 1)
			start_index = random.randint(0, len(text) - maxlen - 1)
			generated_text = text[start_index: start_index + maxlen]
			#print("maxlen", maxlen, "stindex", start_index, len(generated_text))
			print('--- Generating with seed: "' + generated_text + '"')

			for temperature in [0.1, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.2, 1.5]:
				print("")
				print("")
				print('------ temperature:', temperature)

				sys.stdout.write(generated_text)

				for i in range(400):
					sampled = np.zeros((1, maxlen, len(chars)))
					for t, char in enumerate(generated_text):
						sampled[0, t, char_indices[char]] = 1.

					preds = self.model.predict(sampled, verbose=0)[0]
					next_index = self.sample(preds, temperature)
					next_char = chars[next_index]

					generated_text += next_char
					generated_text = generated_text[1:]

					sys.stdout.write(next_char)
					sys.stdout.flush()


	def reweight_distribution(self, original_distribution, temperature=0.5):
		distribution = np.log(original_distribution) / temperature
		distribution = np.exp(distribution)

		return distribution / np.sum(distribution)

	def test(self):
		pass
	def train(self):
		pass
	def predict(self):
		pass

if __name__ == "__main__":
	nn = deepNiche()