import numpy as np
from Model import Model
import collections


class Random_Model(Model):
    def __init__(self, nb_class, dict_class, mwe_prediction=False):#, seed=0):
        super().__init__()
        #np.random.seed(seed)
        self.nb_class = nb_class
        self.mwe_prediction = mwe_prediction
        self.dict_class = dict_class
        self.nominals = [val for key, val in dict_class.items() if key.startswith('n.')]
        self.verbals = [val for key, val in dict_class.items() if key.startswith('v.')]
    
    def fit(self, X, y):
        pass
    
    def predict(self, X_test):
        y_pred = []
        if self.mwe_prediction:
            y_pred = np.random.choice(list(self.dict_class.values()), size=len(X_test))
        else:
            for xi_test in X_test:
                if xi_test[3] == 'NOUN':
                    y_pred.append(np.random.choice(self.nominals))
                elif xi_test[3] == 'VERB':
                    y_pred.append(np.random.choice(self.verbals))
                else:
                    y_pred.append(self.dict_class[''])
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        if self.mwe_prediction:
            return super().score(X_test, y_test, self.dict_class[''])
        else:
            return super().score(X_test, y_test, self.dict_class[''])


class Random_Distribution_Model(Model):
    def __init__(self, dict_class, mwe_prediction=False):
        super().__init__()
        self.y_pred = 0
        self.dict_class = dict_class
        self.mwe_prediction = mwe_prediction
        self.nominals = [val for key, val in dict_class.items() if key.startswith('n.')]
        self.verbals = [val for key, val in dict_class.items() if key.startswith('v.')]
        self.nominals_dist = None
        self.verbals_dist = None
        self.mwe_dist = None
        self.mwe = None
    
    def fit(self, X, y):
        if self.mwe_prediction:
            self.mwe, self.mwe_dist = np.unique(y, return_counts=True)
            self.mwe_dist = self.mwe_dist/np.sum(self.mwe_dist)
        else:
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
        if self.mwe_prediction:
            return np.random.choice(self.mwe, p=self.mwe_dist, size=len(X_test))
        else:
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
        if self.mwe_prediction:
            return super().score(X_test, y_test, self.dict_class[''])
        else:
            return super().score(X_test, y_test, self.dict_class[''])


class Majority_Class_Selection_Model(Model):
    def __init__(self, dict_class, mwe_prediction=False):
        super().__init__()
        self.y_pred = 0
        self.dict_class = dict_class
        self.mwe_prediction = mwe_prediction
        self.nominals = [val for key, val in dict_class.items() if key.startswith('n.')]
        self.verbals = [val for key, val in dict_class.items() if key.startswith('v.')]
        self.most_nominal = None
        self.most_verbal = None
        self.most_mwe = None
    
    def fit(self, X, y):
        if self.mwe_prediction:
            _, counts = np.unique(y, return_counts=True)
            self.most_mwe = np.argmax(counts, axis=0)
        else:
            y_nominals = []
            y_verbals = []
            self.inv_dict_class = {v: k for k, v in self.dict_class.items()}
            for xi, yi in zip(X, y):
                if xi[3] == 'NOUN' and self.inv_dict_class[yi].startswith('n.'):
                    y_nominals.append(yi)
                elif xi[3] == 'VERB' and self.inv_dict_class[yi].startswith('v.'):
                    y_verbals.append(yi)
            
            self.most_nominal = collections.Counter(y_nominals).most_common(1)[0][0]
            self.most_verbal = collections.Counter(y_verbals).most_common(1)[0][0]
    
    def predict(self, X_test):
        y_pred = []
        if self.mwe_prediction:
            for xi_test in X_test:
                y_pred.append(self.most_mwe)
        else:
            for xi_test in X_test:
                if xi_test[3] == 'NOUN':
                    y_pred.append(self.most_nominal)
                elif xi_test[3] == 'VERB':
                    y_pred.append(self.most_verbal)
                else:
                    y_pred.append(self.dict_class[''])
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        if self.mwe_prediction:
            return super().score(X_test, y_test, self.dict_class[''])
        else:
            return super().score(X_test, y_test, self.dict_class[''])




def extract_data(dict_class, data, extract_mwe=False, is_test_blind=False):
    X, y = [], [] #np.empty((len(data), 4), dtype=object), np.empty((len(data),), dtype=np.int)
    for i, line in enumerate(data):
        if line != '\n':
            # Don't know what is the 6th and 9th element, the 7th is always empty
            num, word, lem, pos, mwe, _, _, supersenses, _ = line[:-1].split('\t')
            X.append( np.array([int(num), word, lem, pos], dtype=object) )
            if extract_mwe:
                y.append( dict_class[mwe] )
            else:
                y.append( dict_class[supersenses] )
    
    return np.array(X, dtype=object), np.array(y, dtype=int)


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
    dict_sst = collections.defaultdict(lambda: len(dict_sst))
    dict_mwe = collections.defaultdict(lambda: len(dict_mwe))
    
    data = open('../dimsum-data-1.5/dimsum16.train', 'r').readlines()
    
    """
    X : num, word, lem, POS
    y_sst : class (in 26 noun supersenses + 15 verb supersenses)
    y_mwe : class ('B', 'I', 'O', 'b', 'i', 'o')
    """
    
    X, y_sst = extract_data(dict_sst, data)
    X, y_mwe = extract_data(dict_mwe, data, extract_mwe=True)
    X_, X_val, y_sst, y_val_sst = train_test_split(X, y_sst, test_size=0.3, random_state=0, shuffle=True)
    X, X_val, y_mwe, y_val_mwe = train_test_split(X, y_mwe, test_size=0.3, random_state=0, shuffle=True)
    
    #print('X shape', X.shape, ', X_val shape', X_val.shape)
    #print('5 first input (X):', X[:5])
    #print('5 first sst output (y):', y_sst[:5])
    #print('5 first mwe output (y):', y_mwe[:5])
    print(dict_mwe)
    nb_sst = len(dict_sst.values())
    nb_mwe = len(dict_mwe.values())
    #print(dict_class)
    
    rand_model_sst = Random_Model(nb_sst, dict_sst)
    rand_model_mwe = Random_Model(nb_mwe, dict_mwe, mwe_prediction=True)
    #model.fit(X, y)
    #print('5 first prediction:', rand_model_mwe.predict(X[:5]))
    
    #print(dict_sst)
    print('nb_sst:', nb_sst) # Should be equal to 41 + 1 when there is no supersense
    print('nb_mwe:', nb_mwe) # Should be equal to 6
    
    precision, recall, f1_score, accuracy = rand_model_sst.score(X_val, y_val_sst)
    print(f'Random Class Selection model score for SST: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
    precision, recall, f1_score, accuracy = rand_model_mwe.score(X_val, y_val_mwe)
    print(f'Random Class Selection model score for MWE: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
    
    
    rand_dist_model_sst = Random_Distribution_Model(dict_sst)
    rand_dist_model_mwe = Random_Distribution_Model(dict_mwe, mwe_prediction=True)
    rand_dist_model_sst.fit(X, y_sst)
    rand_dist_model_mwe.fit(X, y_mwe)
    #print('5 first prediction:', rand_dist_model.predict(X[:5]))
    precision, recall, f1_score, accuracy = rand_dist_model_sst.score(X_val, y_val_sst)
    print(f'Random Distribution Class Selection model score for SST: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
    precision, recall, f1_score, accuracy = rand_dist_model_mwe.score(X_val, y_val_mwe)
    print(f'Random Distribution Class Selection model score for MWE: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
    
    
    maj_model_sst = Majority_Class_Selection_Model(dict_sst)
    maj_model_mwe = Majority_Class_Selection_Model(dict_mwe, mwe_prediction=True)
    maj_model_sst.fit(X, y_sst)
    maj_model_mwe.fit(X, y_mwe)
    #print('5 first prediction:', maj_model.predict(X[:5]))
    precision, recall, f1_score, accuracy = maj_model_sst.score(X_val, y_val_sst)
    print(f'Majority Class Selection model score for SST: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
    precision, recall, f1_score, accuracy = maj_model_mwe.score(X_val, y_val_mwe)
    print(f'Majority Class Selection model score for MWE: Acc={accuracy:.4} P={precision:.4f}, R={recall:.4f}, F1={f1_score:.4f}')
    
    
    
    #write_data(X_val, y_val, dict_class, 'val.gold')
    #write_data(X_val, rand_model.predict(X_val), dict_class, 'rand_class_val.pred')
    #write_data(X_val, maj_model.predict(X_val), dict_class, 'maj_class_val.pred')
    #write_data(X_val, rand_dist_model.predict(X_val), dict_class, 'rand_dist_class_val.pred')
    
    # Command to compute scores with dimsumeval: python dimsumeval.py -C ../../models/val.gold ../../models/maj_class_val.pred

