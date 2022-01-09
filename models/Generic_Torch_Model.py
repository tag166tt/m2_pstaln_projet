import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

import collections
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


def train_test_split_sentences(X, y, test_size=0.3, random_state=0, shuffle=True) :
    X_by_sentences = []
    y_by_sentences = []
    sentence = []
    sentence_tags = []
    for line, tag in zip(X, y):
        if line == '':
            if len(sentence) != 0:
                sentence.append([sentence[-1][0]+1, '<eos>', '', ''])
                sentence_tags.append('<eos>')
                X_by_sentences.append(sentence)
                y_by_sentences.append(sentence_tags)
                sentence = []
                sentence_tags = []
        else:
            sentence.append(line)
            sentence_tags.append(tag)

    X, X_val, y, y_val = train_test_split(X_by_sentences, y_by_sentences, test_size=test_size, random_state=random_state, shuffle=shuffle)

    X = [y for x in X for y in x]
    X_val = [y for x in X_val for y in x]
    y = [y_1 for x in y for y_1 in x]
    y_val = [y for x in y_val for y in x]

    return X, X_val, y, y_val


def write_data(X_test, y_test, filename):
    #xi_test format: num, word, lem, POS
    with open(filename, 'w', encoding='utf-8') as f:
        is_first = True
        for xi_test, yi_test in zip(X_test, y_test):
            if (xi_test[1] == '<eos>'):# or (yi_test == '<eos>'):
                continue
            if (not is_first) and xi_test[0] == 1:
                f.write('\n')
            yi_test = yi_test if yi_test != '<eos>' else ''
            f.write(f'{xi_test[0]}\t{xi_test[1]}\t-\t{xi_test[3]}\tO\t0\t\t{yi_test}\n')
            is_first = False
        f.write('\n')


def extract_data(data_in):
    X, y = [], []
    for i, line in enumerate(data_in):
        if line != '\n':
            num, word, lem, pos, mwe, _, _, supersenses, _ = line[:-1].split('\t')
            X.append([int(num), word, lem, pos])
            y.append(supersenses)
        else :
            X.append('')
            y.append('')
    return X, y


def encode_sentence(le_vocab, int_texts, int_labels, max_len, use_pos):
    temp_x, temp_y, final_y, final_x = [], [], [], []
    eos_symbol = le_vocab['<eos>']
    for x, y in zip(int_texts, int_labels):
        temp_x.append(x)
        temp_y.append(y)
        if (use_pos and (x[0] == eos_symbol)) or (not use_pos and (x == eos_symbol)):
            final_x.append(temp_x)
            final_y.append(temp_y)
            temp_x = []
            temp_y = []

    if use_pos:
        X = torch.zeros(len(final_x), max_len, 2).long()
    else:
        X = torch.zeros(len(final_x), max_len).long()

    Y = torch.zeros(len(final_y), max_len).long()
    for i, (text, label) in enumerate(zip(final_x, final_y)):
        length = min(max_len, len(text))
        X[i,:length] = torch.LongTensor(text[:length])
        Y[i,:length] = torch.LongTensor(label[:length])

    #print(len(int_texts))
    #print(len(final_x))
    #print(X.size())

    return X, Y


def transform(X, y, X_val, y_val, max_len=16, batch_size=64, use_pos=False):
    le_pos = collections.defaultdict(lambda: len(le_pos))
    le = collections.defaultdict(lambda: len(le))
    le['<eos>'] = 0
    le_vocab = collections.defaultdict(lambda: len(le_vocab))
    le_vocab['<eos>'] = 0
    
    int_labels = [le[yi] for yi in y]
    int_labels_val = [le[yi] for yi in y_val]
    
    if use_pos:
        vocab = [[text, pos] for _, text, _, pos in X]
        vocab_val = [[text, pos] for _, text, _, pos in X_val]
        int_texts = [[le_vocab[word], le_pos[pos]] for word, pos in vocab]
        int_texts_val = [[le_vocab[word], le_pos[pos]] for word, pos in vocab_val]
    else:
        vocab = [text for _, text, _, _ in X]
        vocab_val = [text for _, text, _, _ in X_val]
        int_texts = [le_vocab[word] for word in vocab]
        int_texts_val = [le_vocab[word] for word in vocab_val]
    
    X_train, Y_train = encode_sentence(le_vocab, int_texts, int_labels, max_len, use_pos)
    X_valid, Y_valid = encode_sentence(le_vocab, int_texts_val, int_labels_val, max_len, use_pos)
    
    train_set = TensorDataset(X_train, Y_train)
    valid_set = TensorDataset(X_valid, Y_valid)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    
    if use_pos:
        return vocab, le_vocab, le_pos, le, train_loader, valid_loader
    else:
        return vocab, le_vocab, le, train_loader, valid_loader


def load_pretrained_weights(filename, vocab_size=None, le_vocab=None, from_pickle=False, dim=300):
    """
    from_pickle: 'pretrained_weights.pkl'
    from_vec: 'wiki-news-300d-1M.vec'
    """
    if from_pickle:
        with open(filename, 'rb') as fin:
            pretrained_weights = pickle.load(fin)
    else: # from_vec
        pretrained_weights = torch.zeros(vocab_size, dim)
        with open(filename, encoding="utf-8") as fp:
            fp.readline()
            for line in fp:
                tokens = line.strip().split()
                if tokens[0].lower() in le_vocab:
                    pretrained_weights[le_vocab[tokens[0].lower()]] = torch.FloatTensor([float(x) for x in tokens[1:]])
    return pretrained_weights


def transform_test(le_vocab, le, X, y=None, max_len=16, le_pos=None):
    use_pos = (le_pos is not None)
    if use_pos:
        vocab = [[text, pos] for _, text, _, pos in X]
        int_texts = [[le_vocab[word], le_pos[pos]] for word, pos in vocab]
    else:
        vocab = [text for _, text, _, _ in X]
        int_texts = [le_vocab[word] for word in vocab]
    int_labels = [le[yi] for yi in y]
    
    X_test, Y_test = encode_sentence(le_vocab, int_texts, int_labels, max_len, use_pos)#encode_sentence(int_texts, int_labels)
    
    return X_test, Y_test


def align_pred(X_orig, y_pred, le):
    y_trf_pred = []
    i_y = 0
    i = 0
    max_i = y_pred.size(1)
    is_first = True
    
    for xi in X_orig:
        if (not is_first) and xi[0] == 1:
            i_y += 1
            i = 0
        
        if i >= max_i:
            next_pred = le['']
        else:
            next_pred = y_pred[i_y][i].item()
        
        y_trf_pred.append( next_pred )
        i += 1
        is_first = False
    return y_trf_pred


from Model import Model

class Simple_GRU(nn.Module):
    def __init__(self, pretrained_weights, le, embed_size=300, hidden_size=128, use_pos=False):
        super().__init__()
        self.use_pos = use_pos
        if self.use_pos:
            self.embed = nn.Embedding(len(le), embed_size, padding_idx=le['<eos>'])
            self.embed.weight = nn.Parameter(pretrained_weights, requires_grad=False)
            self.rnn = nn.GRU(embed_size+1, hidden_size, bias=False, num_layers=1, bidirectional=False, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.decision = nn.Linear(hidden_size * 1 * 1, len(le))
        else:
            self.embed = nn.Embedding(len(le), embed_size, padding_idx=le['<eos>'])
            self.embed.weight = nn.Parameter(pretrained_weights, requires_grad=False)
            self.rnn = nn.GRU(embed_size, hidden_size, bias=False, num_layers=1, bidirectional=False, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.decision = nn.Linear(hidden_size * 1 * 1, len(le))

    def forward(self, x):
        if self.use_pos:
            embed = self.embed(x[:, :, 0])
            concat = torch.cat((x[:, :, 1].view(x.size(0), x.size(1), 1), embed), dim=2)
            output, hidden = self.rnn(concat)
        else:
            embed = self.embed(x)
            output, hidden = self.rnn(embed)
        return self.decision(self.dropout(output))



class NN_Model(Model): # Compatible with torch model only
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), optim=optim.Adam):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optim = optim

    def fit(self, train_loader, valid_loader, epochs):
        optimizer = self.optim(filter(lambda param: param.requires_grad, self.model.parameters()))
        for epoch in range(epochs):
            self.model.train()
            total_loss = num = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                y_scores = self.model(x)
                loss = self.criterion(y_scores.view(y.size(0) * y.size(1), -1), y.view(y.size(0) * y.size(1)))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num += len(y)
            print(epoch+1, total_loss / num, *self.perf(valid_loader))
    
    def perf(self, loader):
        self.model.eval()
        total_loss = correct = num_loss = num_perf = 0
        for x, y in loader:
            with torch.no_grad():
                y_scores = self.model(x)
                loss = self.criterion(y_scores.view(y.size(0) * y.size(1), -1), y.view(y.size(0) * y.size(1)))
                y_pred = torch.max(y_scores, 2)[1]
                mask = (y != 0)
                correct += torch.sum((y_pred.data == y) * mask)
                total_loss += loss.item()
                num_loss += len(y)
                num_perf += torch.sum(mask).item()
        return total_loss / num_loss, correct.item() / num_perf

    def predict(self, X_test):
        self.model.eval()
        return torch.max(self.model(X_test), 2)[1]

    def score(self, X_test, y_test, exception_class, le):
        #return super().score(X_test, y_test, exception_class)
        """
        Returns:
            precision = #(correct)/#(predicted)
            recall = #(correct)/#(gold)
            F1_score = 2*precision*recall / (precision + recall)
            accuracy = #(correct: supersense or no-supersense)/#(tokens)
        """

        y_hat = self.predict(X_test)
        
        y_test = y_test.view((-1,))
        y_hat = y_hat.view((-1,))
        #print(len(y_test))

        nb_predicted, nb_gold, nb_tokens = 0, 0, 0
        nb_correct, accuracy = 0, 0
        for yi, yi_hat in zip(y_test, y_hat):
            yi, yi_hat = yi.item(), yi_hat.item()
            #print(yi_hat, yi)
            if yi != le['<eos>']:
                yi_hat = yi_hat if yi_hat != le['<eos>'] else le['']
                if yi_hat == yi:
                    accuracy += 1
                if yi_hat != exception_class:
                    nb_predicted += 1
                if yi != exception_class:
                    nb_gold += 1
                if (yi != exception_class) and (yi == yi_hat):
                    nb_correct += 1
                nb_tokens += 1

        precision = nb_correct/nb_predicted if nb_predicted != 0 else 0.
        recall = nb_correct/nb_gold if nb_gold != 0 else 0.
        f1_score = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0.
        accuracy = accuracy/nb_tokens
        return precision, recall, f1_score, accuracy

