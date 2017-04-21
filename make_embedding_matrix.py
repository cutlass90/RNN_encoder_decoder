import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from sklearn.preprocessing import OneHotEncoder
import itertools as it
import tensorflow as tf


data = pd.read_csv('dataset/temp.csv')


list_of_tokens = []
for i in range(data.shape[0]):
    list_of_tokens += word_tokenize(data['question1'][i])
    list_of_tokens += word_tokenize(data['question2'][i])
    list_of_tokens = list(set(list_of_tokens))

print(len(list_of_tokens))
 
"""
tokens = word_tokenize(q1[1])

###



uniq_tokens = set(tokens)
right_tokens, strange_tokens = [],[]

for t in uniq_tokens:
    # if t in w2v_model.vocab:
    if True:
        right_tokens.append(t)
    else:
        strange_tokens.append(t)
print('Find {} strange words'.format(len(strange_tokens)))



###




all_tokens = it.chain.from_iterable(list_of_tokens)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokens_docs]

# convert list of token-id lists to one-hot representation
vec = OneHotEncoder(n_values=len(word_to_id))
X = vec.fit_transform(token_ids)

print X.toarray()
"""

"""
em = tf.reshape(tf.range(start=0, limit=50), [10,5])
embed = tf.nn.embedding_lookup(params=em, ids=[0,2,9])
sess = tf.InteractiveSession()

print(sess.run(em))
print(sess.run(embed))
"""