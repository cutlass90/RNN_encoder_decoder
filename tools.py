from string import punctuation
import pdb
import itertools as it
import random

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import gensim
from gensim.models.keyedvectors import KeyedVectors

class EpochFinished(Exception):
    pass

stop_words = [
    'the','a','an','and','but','if','or','because','as','what','which','this',
    'that','these','those','then','just','so','than','such','both','through',
    'about','for','is','of','while','during','to','What','Which','Is','If',
    'While','This']

#-------------------------------------------------------------------------------
def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)







class BatchProvider(object):

    #---------------------------------------------------------------------------
    def __init__(self, data, loader_class):
        self.data = data
        self.loader_class = loader_class
        self.i = 0

    #---------------------------------------------------------------------------
    def sample_batch(self, batch_size):
        """ questions has shape batch_size x max_len x n_features"""
        batch = {}
        data = self.data.sample(batch_size)

        questions, lengths = self.zero_pad(data['question1'])
        batch['seq_lengths_1'] = lengths.astype(np.int32)
        batch['questions_1'] = questions.astype(np.float32)

        questions, lengths = self.zero_pad(data['question2'])
        batch['seq_lengths_2'] = lengths.astype(np.int32)
        batch['questions_2'] = questions.astype(np.float32)

        if 'is_duplicate' in self.data.columns.values:
            batch['targets'] = data['is_duplicate'].as_matrix().astype(np.float32)
        return batch

    #---------------------------------------------------------------------------
    def next_batch(self, batch_size):
        """ questions has shape batch_size x max_len x n_features"""
        data = self.data[self.i:self.i+batch_size]
        if data.shape[0] == 0:
            raise(EpochFinished())
        batch = self.process_batch(data)
        self.i += batch_size
        return batch

    #---------------------------------------------------------------------------
    def process_batch(self, data):
        batch = {}
        questions, lengths = self.zero_pad(data['question1'])
        batch['seq_lengths_1'] = lengths.astype(np.int32)
        batch['questions_1'] = questions.astype(np.float32)

        questions, lengths = self.zero_pad(data['question2'])
        batch['seq_lengths_2'] = lengths.astype(np.int32)
        batch['questions_2'] = questions.astype(np.float32)
        
        if 'is_duplicate' in data.columns.values:
            batch['targets'] = data['is_duplicate'].as_matrix().astype(np.float32)

        return batch

    #---------------------------------------------------------------------------
    def zero_pad(self, sentences):
        """ sentences is iterable part of text """
        list_of_vectorlist = [self.text_to_vectorlist(text) for text in sentences]
        lengths = [len(l) for l in list_of_vectorlist]
        max_len = max(lengths)
        for i in range(len(list_of_vectorlist)):
            while len(list_of_vectorlist[i]) < max_len:
                list_of_vectorlist[i].append(np.zeros([300]))        
        return np.array(list_of_vectorlist), np.array(lengths)

    #---------------------------------------------------------------------------
    def text_to_vectorlist(self, text):
        text = text_to_wordlist(text, remove_stop_words=False)
        list_of_w = text.split(' ')
        list_of_v = [self.loader_class.w2v_model.word_vec(word, use_norm=False)\
            for word in list_of_w if word in self.loader_class.w2v_model.vocab]
        return list_of_v



class DataProvider(object):

    def __init__(self, path_to_csv, path_to_w2v, test_size):
        self.data = pd.read_csv(path_to_csv)
        self.data = self.data.fillna('empty')
        if test_size>0:
            self.train_data, self.test_data = train_test_split(self.data,
            test_size=test_size, random_state=123)
            self.train_data.reset_index(drop=True, inplace=True)
            self.test_data.reset_index(drop=True, inplace=True)
        else:
            self.train_data = self.data
            self.test_data = np.empty([0])
        print('train_data', self.train_data.shape)
        print('test_data', self.test_data.shape)
        self.w2v_model = KeyedVectors.load_word2vec_format(path_to_w2v, binary=True)

        self.train = BatchProvider(data=self.train_data, loader_class=self)
        if test_size > 0:
            self.test = BatchProvider(data=self.test_data, loader_class=self)

    def save_vectors_as_words(self, path_to_file, lists_of_v):
        # lists_of_v is list of list of vectors
        lists_of_w = [[self.w2v_model.most_similar([w], topn=1)[0][0] for w in s]\
            for s in lists_of_v]

        with open(path_to_file, 'w') as f:
            for s in lists_of_w:
                [f.write(w+' ') for w in s]
                f.write('\n')




#-------------------------------------------------------------------------------
if __name__ == '__main__':
    data_provider = DataProvider(path_to_csv='dataset/temp.csv',
        path_to_w2v='embeddings/GoogleNews-vectors-negative300.bin',
        test_size=0.2) 

    """
    for i in range(10):
        print(i)
        data_provider.train.sample_batch(16)
        data_provider.test.sample_batch(16)

    for i in range(10):
        print(i)
        data_provider.train.next_batch(16)
        data_provider.test.next_batch(16)
    """
    batch_size = 2
    batch = data_provider.train.next_batch(batch_size)
    res = batch['questions_1']
    result = []
    result += [[w for w in res[i,:batch['seq_lengths_1'][i],:]]\
    for i in range(batch_size)]
    data_provider.save_vectors_as_words('test.txt', result)

