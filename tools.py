from string import punctuation
import pdb
import itertools as it
import random
import os
import pickle

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
def text_to_wordlist(text, remove_stop_words=False, stem_words=False, lower=True):
    """Clean the text, with the option to remove stop_words and to stem words.

        Inputs:
            text: string object

        Return:
            string of processed text
    """

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
    
    if lower:
        text = text.lower()
    return text







class BatchProvider(object):

    #---------------------------------------------------------------------------
    def __init__(self, data, loader_class):
        self.data = data
        self.loader_class = loader_class
        self.i = 0

    #---------------------------------------------------------------------------
    def sample_batch(self, batch_size):
        """ samples batch from dataset

            Return:
                dict with next fields:
                questions_1 - array of indexes in embedding as_matrix
                    shape is batch_size x max_len
                seq_lengths_1 - array with lenghs of sentences
                    shape is batch_size
        """

        data = self.data.sample(batch_size)
        batch = self.process_batch(data)
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
        text = text_to_wordlist(text)
        list_of_w = text.split(' ')
        list_of_v = [self.loader_class.w2v_model.word_vec(word)\
            for word in list_of_w if word in self.loader_class.w2v_model.vocab]
        return list_of_v



class DataProvider(object):

    def __init__(self, path_to_csv, path_to_w2v, test_size, path_to_vocab):
        self.data = pd.read_csv(path_to_csv)
        self.data = self.data.fillna('empty')
        if os.path.isfile(path_to_vocab):
            print('Load vocab', end='...')
            with open('dataset/vocab.pickle', 'rb') as f:
                self.vocab = pickle.load(f)
            print('len of vocab = {}'.format(len(self.vocab)), end='  ')
            print('Done')
        else:
            print('Processing vocab', end='...')
            self.vocab = self.get_vocab(self.data['question1'])
            with open('dataset/vocab.pickle', 'wb') as f:
                pickle.dump(self.vocab, f)
            print('len of vocab = {}'.format(len(self.vocab)), end='  ')
            print('Done')
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
        print('Load vord to vec', end='...')
        self.w2v_model = KeyedVectors.load_word2vec_format(path_to_w2v, binary=True)
        print('Done')


        self.train = BatchProvider(data=self.train_data, loader_class=self)
        if test_size > 0:
            self.test = BatchProvider(data=self.test_data, loader_class=self)

    def save_vectors_as_words(self, path_to_file, lists_of_v):
        """ Save results to txt file

        Args:
            path_to_file: string, file to save
            lists_of_v: list of list of vectors
        """
        lists_of_w = []
        for s in lists_of_v:
            sentence = []
            for v in s:
                most_similar = self.w2v_model.most_similar([v], topn=1000)
                most_similar = [i[0] for i in most_similar]
                for i, w in enumerate(most_similar):
                    if w in self.vocab:
                        sentence.append(w)
                        break
                    if i == 999:
                        print('Not find similar word')
                        sentence.append('not_similar')
                        break
            lists_of_w.append(sentence)

        with open(path_to_file, 'w') as f:
            for s in lists_of_w:
                [f.write(w+' ') for w in s]
                f.write('\n')

    def get_vocab(self, data):
        """ Return list of unique words from data

        Args:
            data: iterable object of sentences (string)

        Returns:
            vocab: list of unique words
        """
        vocab = set()
        for s in data:
            text = text_to_wordlist(s)
            list_of_w = text.split(' ')
            vocab = vocab.union(set(list_of_w))

        return vocab






#-------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    data_provider = DataProvider(path_to_csv='dataset/set_for_encoding.csv',
        path_to_w2v='embeddings/GoogleNews-vectors-negative300.bin',
        test_size=0.2, path_to_vocab='dataset/vocab.pickle') 
    """

    """
    for i in range(10):
        print(i)
        data_provider.train.sample_batch(16)
        data_provider.test.sample_batch(16)

    for i in range(10):
        print(i)
        data_provider.train.next_batch(16)
        data_provider.test.next_batch(16)
    
    batch_size = 2
    batch = data_provider.train.next_batch(batch_size)
    res = batch['questions_1']
    result = []
    result += [[w for w in res[i,:batch['seq_lengths_1'][i],:]]\
    for i in range(batch_size)]
    data_provider.save_vectors_as_words('test.txt', result)
    """

    """
    wordlist = text_to_wordlist('What is the story of Kohinoor (Koh-i-Noor) Diamond?')
    print(wordlist)
    print(type(wordlist))
    """

    # print(len(data_provider.vocab))