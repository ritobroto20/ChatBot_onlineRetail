import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer= PorterStemmer()

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sent, words):
    sentence_words= [stem(word) for word in tokenized_sent]
    bag= np.zeros(len(words), dtype=np.float32)
    for idx,word in enumerate(words):
        if word in sentence_words:
            bag[idx]=1
    return bag

a=[2,3,4,1,6,1]
print(a.index(2),a.index(1))