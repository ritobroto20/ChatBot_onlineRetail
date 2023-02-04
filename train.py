# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

import numpy as np
import json
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.tokenize import word_tokenize
from nltk_utils import stem,bag_of_words

#loading file
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
    tags.append(intent['tag'])
    for sent in intent['patterns']:
        sent_tokenized= word_tokenize(sent)
        all_words.extend(sent_tokenized)
        xy.append((sent_tokenized,intent['tag']))

ignore_words= ['.', ',', '?', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
print(all_words,'\n')

x_train= []
y_train=[]
for (pattern_sentence,tag) in xy:
    stemmed_sent=[]
    for w in pattern_sentence:
        stemmed_sent.append(stem(w))
    bag= bag_of_words(stemmed_sent,all_words)
    x_train.append(bag)
    label= tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyper-parameters
N_epochs= 1000
batch_size= 9
learning_rate= 0.01
input_size= len(x_train[0])
hidden_size= 8
output_size= len(tags)

# Dataset(Pytorch):Link- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples= len(x_train)
        self.x_data= x_train
        self.y_data= y_train

    # support indexing such that dataset[i] can be used to get i-th sample:
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset= ChatDataset()
train_loader= DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

model= NeuralNet(input_size, hidden_size, output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(),lr=learning_rate)

# Training the model
for epoch in range(N_epochs):
    print("yo")
    i=0
    for (words,labels) in train_loader:

        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs= model(words)               # Forward pass
        loss= criterion(outputs,labels)     # loss calculation using- Cross Entropy Loss

        print("husi")
        # BAckward propagation
        optimizer.zero_grad()

        loss.backward()
        print("awsdef")
        optimizer.step()
if (epoch+1) % 100 ==0:
    print(epoch)
    print(f"epoch {epoch}, Loss= {loss.item():.4f}")


print(f"final loss= {loss.item():.4f}")
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
