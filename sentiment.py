
import sys
import io
import os
import re
import numpy as np
import gensim
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from string import punctuation



def read_file(path):
    cwd = os.getcwd()
    data = []
    labels = []

    neg_path = path + "neg"
    for file in os.listdir(neg_path):
        if file.split('.')[1] != 'txt':
            continue
        review_file = io.open(neg_path + '/' + file, 'r')
        file_data = review_file.read()
        file_data = re.sub(r'[^A-Za-z\s]+','',file_data)
        #file_data = list(file_data)
        data.append(file_data)
        labels.append(0)
        review_file.close()

    pos_path = path + "pos"
    for file in os.listdir(pos_path):
        if file.split('.')[1] != 'txt':
            continue
        review_file = io.open(pos_path + '/' + file)
        file_data = review_file.read()
        file_data = re.sub(r'[^A-Za-z\s]+','',file_data)
        #file_data = list(file_data)
        data.append(file_data)
        labels.append(1)
        review_file.close()


    return data, labels


def preprocess(text):

    vocab_dict = {}
    key = 1
    if type(text) is not list:
        sys.exit("Please provide a list to the method")

    for sentences in text:
        words = sentences.split()
        for word in words:
            temp_dict = {}
            if word not in vocab_dict:
                temp_dict = {word:key}
                vocab_dict.update(temp_dict)
                key += 1

    return vocab_dict

def encode_review(vocab, text):

    encoded_list = []

    if type(vocab) is not dict or type(text) is not list:
        sys.exit("Please provide a list to the method")

    for sentence in text:
        temp_list = []
        words = sentence.split()
        for word in words:
            temp_list.append(vocab[word])

        encoded_list.append(temp_list)

    return encoded_list


def pad_zeros(encoded_reviews, seq_length):


    if type(encoded_reviews) is not list:
        sys.exit("Please provide a list to the method")

    features = np.zeros((len(encoded_reviews), seq_length), dtype = int)
    for e in encoded_reviews:
        index = encoded_reviews.index(e)
        length = len(e)
        if length < seq_length:
            zero_list = list(np.zeros(seq_length - length))
            new = e + zero_list


        else:
            new = e[0:seq_length]

        features[index,:] = np.array(new)
    return features


def load_embedding_file(embedding_file, token_dict):

    if not os.path.isfile(embedding_file):
        sys.exit("Input embedding path is not a file")
    if type(token_dict) is not dict:
        sys.exit("Input a dictionary!")
    tensor_dict = {}
    zero_tensor = [0.0] * 300

    temp_dict = {0 : zero_tensor}
    tensor_dict.update(temp_dict)


    vec_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)

    for word in token_dict:
        temp_dict = {}
        if word in vec_model:
            temp_dict = {token_dict[word]: vec_model[word]}
            tensor_dict.update(temp_dict)
        else:
            temp_dict = {token_dict[word]: zero_tensor}
            tensor_dict.update(temp_dict)

    return tensor_dict




def create_data_loader(train_x, train_y, test_x, test_y, batch_size):


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    #Creating Tensor Datasetlabels
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    #shuffling dataset
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


    return train_loader, test_loader


def get_emb(embedding, word):
    emb_result = np.zeros(shape=(25, 200, 300))  # batch_size, pad_seq_length,embed_vec_dim

    for i in range(25):  # batch-size
        for j in range(200):
            emb_result[i, j, :] = embedding[int(word[i, j])]

    emb_result = torch.from_numpy(emb_result).double()
    return emb_result


#Paths
train_path = "movie_reviews/movie_reviews/train/"
test_path = "movie_reviews/movie_reviews/test/"
word2vec_path = "wiki-news-300d-1M.vec"

#Training Data
train_data, train_labels = read_file(train_path)
print("Got Data and Labels")
train_vocab_dict = preprocess(train_data)
print("Got vocabulary")
train_encoded_list = encode_review(train_vocab_dict, train_data)
print("Got encoded list")
train_padded_list = pad_zeros(train_encoded_list, 200)
print("Got Padded List")
train_tensor_dict = load_embedding_file(word2vec_path, train_vocab_dict)
print("Got embedding")

#Test Data
test_data, test_labels = read_file(test_path)
test_vocab_dict = preprocess(test_data)
test_encoded_list = encode_review(test_vocab_dict, test_data)
test_padded_list = pad_zeros(test_encoded_list, 200)
#test_tensor_dict = load_embedding_file(word2vec_path, test_vocab_dict)
print("Test Data Done")

#Creating Tensor Loader
train_loader, test_loader = create_data_loader(train_padded_list, train_labels,test_padded_list, test_labels, 25)


class BaseSentiment(nn.Module):
    def __init__(self, vocab_size,embedding_dim,hidden_dim,output_size):
        super(BaseSentiment, self).__init__()

        self.emd_dict = train_tensor_dict
        self.em = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward (self, input_words):

        op=get_emb(self.emd_dict, input_words)
        op=self.fc(op)
        op=self.sigmoid(op)

        op = op.view(25,-1)
        op = op[:,-1]

        return op

def base_mod_exec(vocab_size,embedding_dim,hidden_dim,output_size,train_loader,test_loader):


    print("Running Baseline Module")
    learning_rate = 0.1

    model = BaseSentiment(vocab_size,embedding_dim,hidden_dim,output_size)

    criterion = nn.BCELoss() 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 20
    print_every=100
    counter=0

    model=model.double()
    model=model.train()

    for e in range(epochs):
        print("Epoch no:",e)
        for inputs,labels in train_loader:
            counter+=1
            output=model(inputs.double())

            
            loss= criterion(output.squeeze(), labels.double())
            loss.backward()
            optimizer.step()


    #Testing
    test_losses=[]
    num_correct=0

    model=model.double()
    model=model.eval()

    for ip,lbl in test_loader:
 
        op=model(ip)

  
        test_loss=criterion(op.squeeze(),lbl.double())
        test_losses.append(test_loss.item())

       
        pred=torch.round(op.squeeze())

        correct_tensor = pred.eq(labels.double().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)


    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))



# Baseline Model
output_size = 1
embedding_dim = 300
hidden_dim = 256
vocab_size = len(train_vocab_dict) + 1

print("Runnind Training and Testing for Baseline Module")

base_mod_exec(vocab_size,embedding_dim,hidden_dim,output_size,train_loader, test_loader)



##################### RNN Module ########################

class RNNSentiment(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_size,n_layers,drop_prob=0.7):
        super(RNNSentiment,self).__init__()

        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim

        self.emd_dict = train_tensor_dict
        self.em=nn.Embedding(vocab_size,embedding_dim)

        #LSTM
        print("LSTM")
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=False)
  

        # #GRU
        # print("GRU")
        # self.gru= nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,bidirectional=True)

        # #Vanilla RNN
        # print("Vanilla RNN")
        # self.rnn= nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,bidirectional=True)

        self.dropout = nn.Dropout(0.7)

        self.fc=nn.Linear(hidden_dim,output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_words,hidden):
        batch_size=25

        embeds=get_emb(self.emd_dict,input_words)  

        #LSTM
        out, hidden = self.lstm(embeds, hidden)
        out = out.contiguous().view(-1, self.hidden_dim) 
        out = self.dropout(out) 

        # #GRU
        # out,hidden = self.gru(embeds)
        # out = out.contiguous().view(-1, self.hidden_dim)
        # out = self.fc(out)
        
        # #Vanilla RNN
        # out,hidden = self.rnn(embeds)
        # out = out.contiguous().view(-1, self.hidden_dim)
        # out = self.fc(out)

        # sigmoid function
        sig_out = self.sigmoid(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden




def rnn_exec(vocab_size,embedding_dim,hidden_dim,output_size,train_loader,test_loader, n_layers,drop_prob):
    


    #Instantiate the model 
    model = RNNSentiment(vocab_size,embedding_dim,hidden_dim,output_size,n_layers,drop_prob)

    # Define loss and optimizer 
    criterion = nn.BCELoss()  #Loss function
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Optimizer 

    epochs = 10
    print_every=50
    counter=0
    clip=5 # gradient clipping 
    batch_size=25

    model=model.double()
    model=model.train()

    
    for epoch in range(epochs):
        print("Epoch no:",epoch)
        h=model.init_hidden(batch_size)

        for inputs,labels in train_loader:
            counter+=1

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            model.zero_grad()

            output,h=model(inputs.double(),h)

            # print(labels.double().size())
            #loss calculation and back propogation
            loss= criterion(output.squeeze(), labels.double())
            loss.backward()
            optimizer.step()

    train_time2=time.time() - start_time2

    #Testing 
    test_losses=[]
    num_correct=0 
    
    h=model.init_hidden(batch_size)

    model=model.double()
    model=model.eval()


    for ip,lbl in test_loader:
        
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # get predicted outputs
        op,h=model(ip,h)

        # calculate loss
        test_loss=criterion(op.squeeze(),lbl.double())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred=torch.round(op.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.double().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)  

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

  

num_layers=2
drop_prob=0.5

print("Running Training and Testing for RNN")
rnn_exec(vocab_size,embedding_dim,hidden_dim,output_size,train_loader, test_loader,num_layers,drop_prob)


########################Self Attention Module##########################

class AttentionSentiment(nn.Module):
    def __init__(self,vocab_size,embedding_dim,output_size):
        super(AttentionSentiment,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.output_size=output_size

        self.emd_dict = train_tensor_dict
        self.em=nn.Embedding(vocab_size,embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads = 1)

        self.fc=nn.Linear(embedding_dim,output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_words):
        batch_size=25

        x=get_emb(self.emd_dict,input_words)  
        x=self.attn(x,x,x)[0]
        x=self.fc(x)
        x=self.sigmoid(x)

        x = x.view(25,-1)
        x = x[:,-1]

        return x

def self_attention_exec (vocab_size,embedding_dim,output_size,train_loader,test_loader):
    #Instantiate the model 
    model = AttentionSentiment(vocab_size,embedding_dim,output_size)

    # Define loss and optimizer 
    criterion = nn.BCELoss()  #Loss function
    learning_rate=0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Optimizer 

    epochs = 20
    print_every=50
    counter=0

    model=model.double()
    model=model.train()

    for epoch in range(epochs):
        print("Epoch no:",epoch)
        for inputs,labels in train_loader:
            counter+=1
            output=model(inputs.double())

            # print(labels.double().size())
            #loss calculation and back propogation
            loss= criterion(output.squeeze(), labels.double())
            loss.backward()
            optimizer.step()


    #Testing 
    test_losses=[]
    num_correct=0 

    model=model.double()
    model=model.eval()

    for ip,lbl in test_loader:
        # get predicted outputs
        op=model(ip)

        # calculate loss
        test_loss=criterion(op.squeeze(),lbl.double())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred=torch.round(op.squeeze())

        # compare predictions to true label
        correct_tensor = pred.eq(labels.double().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

 


print("Running Self Attention Module")
self_attention_exec(vocab_size,embedding_dim,output_size,train_loader, test_loader)

############### CNN ##########################

class CNNSentiment(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_size,n_layers,drop_prob=0.7):
        super(RNNSentiment,self).__init__()

        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim

        self.emd_dict = train_tensor_dict
        self.em=nn.Embedding(vocab_size,embedding_dim)

        self.cnn = nn.CNN(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=False)
  

        self.dropout = nn.Dropout(0.7)

        self.fc=nn.Linear(hidden_dim,output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_words,hidden):
        batch_size=25

        embeds=get_emb(self.emd_dict,input_words)  
        #LSTM
        out, hidden = self.cnn(embeds, hidden)
        out = out.contiguous().view(-1, self.hidden_dim) 
        out = self.dropout(out) 


        # sigmoid function
        sig_out = self.sigmoid(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden




def cnn_exec(vocab_size,embedding_dim,hidden_dim,output_size,train_loader,test_loader, n_layers,drop_prob):
    

    #Instantiate the model 
    model = RNNSentiment(vocab_size,embedding_dim,hidden_dim,output_size,n_layers,drop_prob)

    # Define loss and optimizer 
    criterion = nn.BCELoss()  #Loss function
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Optimizer 

    epochs = 10
    print_every=50
    counter=0
    clip=5 # gradient clipping 
    batch_size=25

    model=model.double()
    model=model.train()

    
    for epoch in range(epochs):
        print("Epoch no:",epoch)
        h=model.init_hidden(batch_size)

        for inputs,labels in train_loader:
            counter+=1

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            model.zero_grad()

            output,h=model(inputs.double(),h)

            # print(labels.double().size())
            #loss calculation and back propogation
            loss= criterion(output.squeeze(), labels.double())
            loss.backward()
            optimizer.step()


    #Testing 
    test_losses=[]
    num_correct=0 
    
    h=model.init_hidden(batch_size)

    model=model.double()
    model=model.eval()


    for ip,lbl in test_loader:
        
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # get predicted outputs
        op,h=model(ip,h)

        # calculate loss
        test_loss=criterion(op.squeeze(),lbl.double())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred=torch.round(op.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.double().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)  

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

 
num_layers=2
drop_prob=0.5

print("Running Training and Testing for CNN")
cnn_exec(vocab_size,embedding_dim,hidden_dim,output_size,train_loader, test_loader,num_layers,drop_prob)
