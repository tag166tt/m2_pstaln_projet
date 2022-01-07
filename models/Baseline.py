import numpy as np
from Model import Model


class Random_Model(Model):
	def __init__(self, nb_class, dict_class, seed=0):
		super().__init__()
		np.random.seed(seed)
		self.nb_class = nb_class
		self.dict_class = dict_class
		self.nominals = [val for key, val in dict_class.items() if key.startswith('n.')]
		self.verbals = [val for key, val in dict_class.items() if key.startswith('v.')]
	
	def fit(self, X, y):
		pass
	
	def predict(self, X_test):
		y_pred = []
		for xi_test in X_test:
			if xi_test[3] == 'NOUN':
				y_pred.append(np.random.choice(self.nominals))
			elif xi_test[3] == 'VERB':
				y_pred.append(np.random.choice(self.verbals))
			else:
				y_pred.append(self.dict_class[''])
		return np.array(y_pred)
	
	def score(self, X_test, y_test):
		return super().score(X_test, y_test, self.dict_class[''])


class Random_Distribution_Model(Model):
	def __init__(self, dict_class):
		super().__init__()
		self.y_pred = 0
		self.dict_class = dict_class
		self.nominals = [val for key, val in dict_class.items() if key.startswith('n.')]
		self.verbals = [val for key, val in dict_class.items() if key.startswith('v.')]
		self.nominals_dist = None
		self.verbals_dist = None
	
	def fit(self, X, y):
		y_nominals = []
		y_verbals = []
		for xi, yi in zip(X, y):
			if xi[3] == 'NOUN':
				y_nominals.append(yi)
			elif xi[3] == 'VERB':
				y_verbals.append(yi)
		
		self.nominals, self.nominals_dist = np.unique(y_nominals, return_counts=True)
		self.nominals_dist = self.nominals_dist/np.sum(self.nominals_dist)
		self.verbals, self.verbals_dist = np.unique(y_verbals, return_counts=True)
		self.verbals_dist = self.verbals_dist/np.sum(self.verbals_dist)
	
	def predict(self, X_test):
		y_pred = []
		for xi_test in X_test:
			if xi_test[3] == 'NOUN':
				y_pred.append(np.random.choice(self.nominals, p=self.nominals_dist))
			elif xi_test[3] == 'VERB':
				y_pred.append(np.random.choice(self.verbals, p=self.verbals_dist))
			else:
				y_pred.append(self.dict_class[''])
		return np.array(y_pred)
	
	def score(self, X_test, y_test):
		return super().score(X_test, y_test, self.dict_class[''])


class Majority_Class_Selection_Model(Model):
	def __init__(self, dict_class):
		super().__init__()
		self.y_pred = 0
		self.dict_class = dict_class
		self.nominals = [val for key, val in dict_class.items() if key.startswith('n.')]
		self.verbals = [val for key, val in dict_class.items() if key.startswith('v.')]
		self.most_nominal = None
		self.most_verbal = None
	
	def fit(self, X, y):
		y_nominals = []
		y_verbals = []
		for xi, yi in zip(X, y):
			if xi[3] == 'NOUN':
				y_nominals.append(yi)
			elif xi[3] == 'VERB':
				y_verbals.append(yi)
		
		_, counts = np.unique(y_nominals, return_counts=True)
		self.most_nominal = np.argmax(counts, axis=0)
		_, counts = np.unique(y_verbals, return_counts=True)
		self.most_verbal = np.argmax(counts, axis=0)
	
	def predict(self, X_test):
		y_pred = []
		for xi_test in X_test:
			if xi_test[3] == 'NOUN':
				y_pred.append(self.most_nominal)
			elif xi_test[3] == 'VERB':
				y_pred.append(self.most_verbal)
			else:
				y_pred.append(self.dict_class[''])
		return np.array(y_pred)
	
	def score(self, X_test, y_test):
		return super().score(X_test, y_test, self.dict_class[''])




def extract_data(dict_class, data, is_test_blind=False):
	X, y = [], [] #np.empty((len(data), 4), dtype=object), np.empty((len(data),), dtype=np.int)
	for i, line in enumerate(data):
		if line != '\n':
			# Don't know what is the 6th and 9th element, the 7th is always empty
			num, word, lem, pos, mwe, _, _, supersenses, _ = line[:-1].split('\t')
			X.append( np.array([int(num), word, lem, pos], dtype=object) )
			y.append( dict_class[supersenses] )
	
	return np.array(X, dtype=object), np.array(y, dtype=np.int)


def write_data(X_test, y_test, dict_class, filename):
	#xi_test format: num, word, lem, POS
	dict_inverse_class = {num: supersense for supersense, num in dict_class.items()}
	with open(filename, 'w', encoding='utf-8') as f:
		is_first = True
		for xi_test, yi_test in zip(X_test, y_test):
			if (not is_first) and xi_test[0] == 1:
				f.write('\n')
			f.write(f'{xi_test[0]}\t{xi_test[1]}\t-\t{xi_test[3]}\tO\t0\t\t{dict_inverse_class[yi_test]}\n')
			is_first = False
		f.write('\n')


if __name__ == '__main__':
	from sklearn.model_selection import train_test_split
	import collections
	dict_class = collections.defaultdict(lambda: len(dict_class))
	
	data = open('../dimsum-data-1.5/dimsum16.train', 'r').readlines()
	data_test = open('../dimsum-data-1.5/dimsum16.test.blind', 'r', encoding='utf-8').readlines()
	
	"""
	X : num, word, lem, POS
	y : class (in 26 noun supersenses + 15 verb supersenses)
	"""
	
	X, y = extract_data(dict_class, data)
	X, X_val, y, y_val = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
	X_test, _ = extract_data(dict_class, data_test)
	
	print('X shape', X.shape, ', X_val shape', X_val.shape, ', X_test shape', X_test.shape)
	print('5 first input (X):', X[:5])
	print('5 first output (y):', y[:5])
	nb_class = len(dict_class.values())
	#print(dict_class)
	
	rand_model = Random_Model(nb_class, dict_class)
	#model.fit(X, y)
	print('5 first prediction:', rand_model.predict(X[:5]))
	
	#print(dict_class)
	print('nb_class:', nb_class) # Should be equal to 41 + 1 when there is no supersense
	
	precision, recall, f1_score, accuracy = rand_model.score(X_val, y_val)
	print(f'Random Class Selection model score: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}') # accuracy ~ 1/42
	
	
	rand_dist_model = Random_Distribution_Model(dict_class)
	rand_dist_model.fit(X, y)
	#print('5 first prediction:', rand_dist_model.predict(X[:5]))
	precision, recall, f1_score, accuracy = rand_dist_model.score(X_val, y_val)
	print(f'Random Distribution Class Selection model score: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
	
	
	maj_model = Majority_Class_Selection_Model(dict_class)
	maj_model.fit(X, y)
	#print('5 first prediction:', maj_model.predict(X[:5]))
	precision, recall, f1_score, accuracy = maj_model.score(X_val, y_val)
	print(f'Majority Class Selection model score: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
	
	
	
	write_data(X_val, y_val, dict_class, 'val.gold')
	write_data(X_val, rand_model.predict(X_val), dict_class, 'rand_class_val.pred')
	write_data(X_val, maj_model.predict(X_val), dict_class, 'maj_class_val.pred')
	write_data(X_val, rand_dist_model.predict(X_val), dict_class, 'rand_dist_class_val.pred')
	
	# Command to compute scores with dimsumeval: python dimsumeval.py -C ../../models/val.gold ../../models/maj_class_val.pred

