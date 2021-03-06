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
    
    if test_size == 0.:
        return [y for x in X_by_sentences for y in x], [[]], [y_1 for x in y_by_sentences for y_1 in x], [[]]
    
    X, X_val, y, y_val = train_test_split(X_by_sentences, y_by_sentences, test_size=test_size, random_state=random_state, shuffle=shuffle)

    X = [y for x in X for y in x]
    X_val = [y for x in X_val for y in x]
    y = [y_1 for x in y for y_1 in x]
    y_val = [y for x in y_val for y in x]

    return X, X_val, y, y_val


def combine_sst_mwe(X, y_sst, y_mwe):
    new_y_sst, new_y_mwe = y_sst.copy(), y_mwe.copy()
    is_first = True
    last_b = -1
    last_B = -1
    seq_sst_low = []
    seq_mwe_low = []
    nb_low = 0
    seq_sst_upp = []
    seq_mwe_upp = []
    #print(len(X))
    #print(len(y_sst))
    #print(len(y_mwe))
    i = 0
    for xi, yi_sst, yi_mwe in zip(X, y_sst, y_mwe):
        #print(i)
        #if (xi[1] == '<eos>'):# or (yi_test == '<eos>'):
        #    continue
        
        if (not is_first) and (xi[0] == 1):
            seq_sst_low = []
            seq_mwe_low = []
            seq_sst_upp = []
            seq_mwe_upp = []
            last_b = -1
            last_B = -1
            nb_low = 0
        
        #print(i, new_y_mwe, nb_low, seq_sst_low, seq_mwe_upp)
        #print(xi)
        
        #print(yi_mwe)
        if yi_mwe in ['<eos>', '', 'O'] or (xi[1] == '<eos>'):
            if (xi[1] == '<eos>'):
                new_y_mwe[i] = '<eos>'
            #print(nb_low)
            #print(yi_mwe, seq_mwe_upp)
            prec_mwe = new_y_mwe[i-1]
            
            if len(seq_sst_upp) > 1:
                new_y_mwe[last_B] = 'B'
                #print(i, seq_mwe_upp, seq_mwe_low, new_y_mwe)
                n = 0
                for j in range(1, len(seq_sst_upp)+nb_low):
                    if (nb_low == 0) or ((nb_low > 0) and (not new_y_mwe[last_B+j].islower())):
                        new_y_mwe[last_B+j] = 'I'
                        new_y_sst[last_B+j] = ''
                        if(seq_sst_upp[0] == ''):
                            #print(j, n, nb_low, seq_sst_upp)
                            try:
                                seq_sst_upp[0] = seq_sst_upp[j-n]
                            except:
                                seq_sst_upp[0] = ''
                    else:
                        n += 1
                new_y_sst[last_B] = seq_sst_upp[0]
                for j in range(1, len(seq_sst_upp)+nb_low):
                    if (nb_low == 0) or ((nb_low > 0) and (not new_y_mwe[last_B+j].islower())):
                        new_y_sst[last_B+j] = ''
            elif (len(seq_sst_upp) == 1):
                new_y_mwe[i-1] = 'O'
            
            if len(seq_sst_low) > 1:
                new_y_mwe[last_b] = 'b'
                for j in range(1, len(seq_sst_low)):
                    new_y_mwe[last_b+j] = 'i'
                    if(seq_sst_low[0] == ''):
                        seq_sst_low[0] = seq_sst_low[j]
                new_y_sst[last_b] = seq_sst_low[0]
                for j in range(1, len(seq_sst_low)):
                    new_y_sst[last_b+j] = ''
            elif (len(seq_sst_low) == 1):
                new_y_mwe[i-1] = 'O'
            
            #print(nb_low)
            if (nb_low > 0) and prec_mwe.islower():# or (len(seq_sst_low) >= 1)
                for j in range(nb_low):
                    if new_y_mwe[i-j-1].islower():
                        new_y_mwe[i-j-1] = 'O'
                if (len(seq_sst_upp) == 1):
                    new_y_mwe[last_B] = 'O'
            
            seq_sst_low = []
            seq_mwe_low = []
            seq_sst_upp = []
            seq_mwe_upp = []
            last_b = -1
            last_B = -1
            nb_low = 0
        elif yi_mwe == 'B':
            if (last_b != -1):
                if len(seq_sst_low) > 1:
                    new_y_mwe[last_b] = 'b'
                    for j in range(1, len(seq_sst_low)):
                        new_y_mwe[last_b+j] = 'i'
                        if(seq_sst_low[0] == ''):
                            seq_sst_low[0] = seq_sst_low[j]
                    new_y_sst[last_b] = seq_sst_low[0]
                    for j in range(1, len(seq_sst_low)):
                        new_y_sst[last_b+j] = ''
                elif (len(seq_sst_low) == 1):
                    new_y_mwe[i-1] = 'o'
                seq_sst_low = []
                seq_mwe_low = []
            
            if (nb_low > 0) and (len(seq_sst_upp) == 0):
                for j in range(nb_low):
                    if new_y_mwe[i-j-1].islower():
                        new_y_mwe[i-j-1] = 'O'
            
            if (last_B != -1) and (nb_low == 0):
                if len(seq_sst_upp) > 1:
                    new_y_mwe[last_B] = 'B'
                    for j in range(1, len(seq_sst_upp)):
                        new_y_mwe[last_B+j] = 'I'
                        if(seq_sst_upp[0] == ''):
                            seq_sst_upp[0] = seq_sst_upp[j]
                    new_y_sst[last_B] = seq_sst_upp[0]
                    for j in range(1, len(seq_sst_upp)):
                        new_y_sst[last_B+j] = ''
                elif (len(seq_sst_upp) == 1):
                    if not new_y_mwe[i-1].islower():
                        new_y_mwe[i-1] = 'O'
                seq_sst_upp = []
                seq_mwe_upp = []
                last_B = -1
                #new_y_mwe[i] = 'I'
                #yi_mwe = 'I'
            #elif (last_B != -1) and (last_b != -1):
            #    new_y_mwe[i] = 'I'
            #    yi_mwe = 'I'
            
            if (last_B == -1):
                last_B = i
            seq_sst_upp.append(yi_sst)
            seq_mwe_upp.append(yi_mwe)
        elif yi_mwe == 'I':
            if (last_b != -1):
                if len(seq_sst_low) > 1:
                    new_y_mwe[last_b] = 'b'
                    for j in range(1, len(seq_sst_low)):
                        new_y_mwe[last_b+j] = 'i'
                        if(seq_sst_low[0] == ''):
                            seq_sst_low[0] = seq_sst_low[j]
                    new_y_sst[last_b] = seq_sst_low[0]
                    for j in range(1, len(seq_sst_low)):
                        new_y_sst[last_b+j] = ''
                elif (len(seq_sst_low) == 1):
                    new_y_mwe[i-1] = 'o'
                seq_sst_low = []
                seq_mwe_low = []
            
            if (nb_low > 0) and (len(seq_sst_upp) == 0):
                for j in range(nb_low):
                    if new_y_mwe[i-j-1].islower():
                        new_y_mwe[i-j-1] = 'O'
            
            if last_B == -1:
                last_B = i
            seq_sst_upp.append(yi_sst)
            seq_mwe_upp.append(yi_mwe)
        elif yi_mwe == 'b':
            if (last_b != -1):
                if len(seq_sst_low) > 1:
                    new_y_mwe[last_b] = 'b'
                    for j in range(1, len(seq_sst_low)):
                        new_y_mwe[last_b+j] = 'i'
                        if(seq_sst_low[0] == ''):
                            seq_sst_low[0] = seq_sst_low[j]
                    new_y_sst[last_b] = seq_sst_low[0]
                    for j in range(1, len(seq_sst_low)):
                        new_y_sst[last_b+j] = ''
                elif (len(seq_sst_low) == 1):
                    new_y_mwe[i-1] = 'o'
                seq_sst_low = []
                seq_mwe_low = []
            last_b = i
            seq_sst_low.append(yi_sst)
            seq_mwe_low.append(yi_mwe)
            nb_low += 1
        elif yi_mwe == 'i':
            if last_b == -1:
                last_b = i
            seq_sst_low.append(yi_sst)
            seq_mwe_low.append(yi_mwe)
            nb_low += 1
        elif yi_mwe == 'o':
            if len(seq_sst_low) > 1:
                new_y_mwe[last_b] = 'b'
                for j in range(1, len(seq_sst_low)):
                    new_y_mwe[last_b+j] = 'i'
                    if(seq_sst_low[0] == ''):
                        seq_sst_low[0] = seq_sst_low[j]
                new_y_sst[last_b] = seq_sst_low[0]
                for j in range(1, len(seq_sst_low)):
                    new_y_sst[last_b+j] = ''
            elif (len(seq_sst_low) == 1):
                if (last_b == -1):
                    new_y_mwe[i-1] = 'O'
                else:
                    new_y_mwe[i-1] = 'o'
            """
            else:
                #new_y_mwe[i] = 'O'
                
                if len(seq_sst_upp) > 1:
                    new_y_mwe[last_B] = 'B'
                    #print(i, seq_mwe_upp, seq_mwe_low, new_y_mwe)
                    n = 0
                    for j in range(1, len(seq_sst_upp)+nb_low):
                        if (nb_low == 0) or ((nb_low > 0) and (not new_y_mwe[last_B+j].islower())):
                            new_y_mwe[last_B+j] = 'I'
                            new_y_sst[last_B+j] = ''
                            if(seq_sst_upp[0] == ''):
                                seq_sst_upp[0] = seq_sst_upp[j-n]
                        else:
                            n += 1
                    new_y_sst[last_B] = seq_sst_upp[0]
                    for j in range(1, len(seq_sst_upp)+nb_low):
                        if (nb_low == 0) or ((nb_low > 0) and (not new_y_mwe[last_B+j].islower())):
                            new_y_sst[last_B+j] = ''
                elif (len(seq_sst_upp) == 1):
                    new_y_mwe[i-1] = 'O'
                
                last_B = -1
                #last_b = -1
                seq_sst_upp = []
                seq_mwe_upp = []
                #nb_low = -1
            """
                
            nb_low += 1
            seq_sst_low = []
            seq_mwe_low = []
            last_b = -1
        
        is_first = False
        i += 1
    
    return new_y_sst, new_y_mwe


def write_data(X_test, y_sst, y_mwe, filename):
    #xi_test format: num, word, lem, POS
    with open(filename, 'w', encoding='utf-8') as f:
        is_first = True
        num_prev_mwe = 1
        num_prev_mwe_lower = 1
        for xi, yi_sst, yi_mwe in zip(X_test, y_sst, y_mwe):
            if (xi[1] == '<eos>'):# or (yi_test == '<eos>'):
                continue
            if (not is_first) and (xi[0] == 1):
                num_prev_mwe = 1
                num_prev_mwe_lower = 1
                f.write('\n')
            yi_sst_ = yi_sst if yi_sst != '<eos>' else ''
            yi_mwe_ = yi_mwe if (yi_mwe != '<eos>' and yi_mwe != '') else 'O'
            num_mwe = num_prev_mwe if yi_mwe_ == 'I' else 0
            num_mwe = num_prev_mwe_lower if yi_mwe_ == 'i' else num_mwe
            
            try:
                f.write(str(f'{xi[0]}\t{xi[1]}\t-\t{xi[3]}\t{yi_mwe_}\t{num_mwe}\t\t{yi_sst_}\n'.encode('utf-8'), 'cp1252'))
            except (UnicodeDecodeError, UnicodeEncodeError):
                f.write(str(f'{xi[0]}\t---\t-\t{xi[3]}\t{yi_mwe_}\t{num_mwe}\t\t{yi_sst_}\n'.encode('utf-8'), 'cp1252'))
            
            #f.write(f'{xi[0]}\t---\t-\t{xi[3]}\t{yi_mwe_}\t{num_mwe}\t\t{yi_sst_}\n')

            is_first = False
            if not yi_mwe.islower():
                num_prev_mwe = xi[0]
            num_prev_mwe_lower = xi[0]
        f.write('\n')


def extract_data(data_in):
    X, y_mwe, y_sst = [], [], []
    for i, line in enumerate(data_in):
        if line != '\n':
            num, word, lem, pos, mwe, _, _, supersenses, _ = line[:-1].split('\t')
            X.append([int(num), word, lem, pos])
            y_sst.append(supersenses)
            y_mwe.append(mwe)
        else :
            X.append('')
            y_sst.append('')
            y_mwe.append('')
    return X, y_sst, y_mwe


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


def transform(X, y, X_val, y_val, max_len=16, batch_size=64, use_pos=False, use_mwe_pred=False):
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
        return vocab, le_vocab, None, le, train_loader, valid_loader


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


def align_pred(X_orig, y_pred, le, is_mwe):
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
            next_pred = le['O'] if is_mwe else le['']
        else:
            next_pred = y_pred[i_y][i].item()
        
        y_trf_pred.append( next_pred )
        i += 1
        is_first = False
    return y_trf_pred


from Model import Model

class Simple_GRU(nn.Module):
    def __init__(self, pretrained_weights, le, embed_size=300, hidden_size=128, use_pos=False, use_mwe=False):
        super().__init__()
        self.use_pos = use_pos
        self.use_mwe = use_mwe
        
        nb_nn = 0
        if self.use_pos:
            nb_nn += 1
        if self.use_mwe:
            nb_nn += 1
        
        self.embed = nn.Embedding(len(le), embed_size, padding_idx=le['<eos>'])
        self.embed.weight = nn.Parameter(pretrained_weights, requires_grad=False)
        self.rnn = nn.GRU(embed_size+nb_nn, hidden_size, bias=False, num_layers=1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.decision = nn.Linear(hidden_size * 1 * 1, len(le))

    def forward(self, x):
        if self.use_mwe or self.use_pos:
            embed = self.embed(x[:, :, 0])
            concat = torch.cat((x[:, :, 1:].view(x.size(0), x.size(1), -1), embed), dim=2)
            output, hidden = self.rnn(concat)
        else:
            embed = self.embed(x)
            output, hidden = self.rnn(embed)
        return self.decision(self.dropout(output))



class NN_Model(Model): # Compatible with torch models only
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
        return torch.max(self.model(X_test), -1)[1]

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


class NN_CRF_Model(Model): # Compatible with torch+torchCRF models only
    def __init__(self, nn_model, crf_model, optim=optim.Adam):
        super().__init__()
        self.nn_model = nn_model
        self.crf_model = crf_model
        self.optim = optim

    def fit(self, train_loader, valid_loader, epochs):
        all_parameters = list(self.nn_model.parameters()) + list(self.crf_model.parameters())
        optimizer = self.optim(filter(lambda param: param.requires_grad, all_parameters))
        for epoch in range(epochs):
            self.nn_model.train()
            self.crf_model.train()
            total_loss = num = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                mask = (y != 0)
                emissions = self.nn_model(x)
                loss = -self.crf_model(emissions, y, mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num += len(y)
            print(epoch+1, total_loss / num, *self.perf(valid_loader))
    
    def perf(self, loader):
        def num_correct(y_pred, y):
            correct = 0
            for y_pred_current, y_current in zip(y_pred, y):
                for i in range(len(y_pred_current)):
                    if y_pred_current[i] == y_current[i]:
                        correct += 1
            return correct

        self.nn_model.eval()
        self.crf_model.eval()
        loss = num_loss = correct = num_perf = 0
        for x, y in loader:
            with torch.no_grad():
                mask = (y != 0)
                emissions = self.nn_model(x)
                y_pred = self.crf_model.decode(emissions, mask)
                correct += num_correct(y_pred, y)
                num_perf += torch.sum(mask).item()
                loss -= self.crf_model(emissions, y, mask).item()
                num_loss += len(y)
        return loss / num_loss, correct / num_perf

    def predict(self, X_test):
        self.nn_model.eval()
        self.crf_model.eval()
        y_pred = self.nn_model(X_test)
        mask = (torch.max(y_pred, -1)[1] != 0)
        return torch.tensor(self.crf_model.decode(y_pred, mask=mask))

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


