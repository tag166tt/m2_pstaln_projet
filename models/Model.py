# Classe template d'un Model

class Model:
	def __init__(self):
		pass
	
	def fit(self, X, y):
		raise NotImplementedError
	
	def predict(self, X_test):
		raise NotImplementedError
	
	def score(self, X_test, y_test, exception_class):
		"""
		Returns:
			precision = #(correct)/#(predicted)
			recall = #(correct)/#(gold)
			F1_score = 2*precision*recall / (precision + recall)
			accuracy = #(correct: supersense or no-supersense)/#(tokens)
		"""
		
		y_hat = self.predict(X_test)
		nb_predicted, nb_gold, nb_tokens = 0, 0, len(X_test)
		nb_correct, accuracy = 0, 0
		for yi_hat, yi in zip(y_hat, y_test):
			if yi_hat == yi:
				accuracy += 1
			if yi_hat != exception_class:
				nb_predicted += 1
			if yi != exception_class:
				nb_gold += 1
			if (yi != exception_class) and (yi == yi_hat):
				nb_correct += 1
		
		precision = nb_correct/nb_predicted if nb_predicted != 0 else 0.
		recall = nb_correct/nb_gold if nb_gold != 0 else 0.
		f1_score = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0.
		accuracy = accuracy/nb_tokens
		return precision, recall, f1_score, accuracy

