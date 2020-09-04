# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:25:16 2020

@author: Qihang Yang
"""

#### REFERENCE: https://www.machinelearningplus.com/nlp/gensim-tutorial/

import gensim
from gensim import corpora
from pprint import pprint

# How to create a dictionary from a list of sentences?
documents = ["Consider a joint distribution in which E[log2 Y |X = x] = β0 +β1 log2 x. Under this model, doubling the value of x produces a 4β1 multiplicative effect on the expected value of Y given X = x. True False", 
             "If we take the residuals from the regression of y on x1, and plot them versus the residuals from regressing x2 on x1, the slope for the least squares regression line on this plot will be exactly equal to b2, the estimated coefficient of x2 in the multiple regression of y on x1 and x2. True False", 
             "If we take the residuals from the regression of y on x1, and plot them versus the residuals from regressing x2 on x1, the intercept for the least squares regression line on this plot will be exactly equal to zero. True False", 
             "The coefficient of multiple determination, denoted R2, gives the proportionate reduction to the total variation in the response variable Y that is achieved by accounting for the linear association between Y and the predictor variables in a regression. True False"]


# Tokenize(split) the sentences into words
texts = [[text for text in doc.split()] for doc in documents]

# Create dictionary
dictionary = corpora.Dictionary(texts)

# Get information about the dictionary
print(dictionary)
#> Dictionary(33 unique tokens: ['Saudis', 'The', 'a', 'acknowledge', 'are']...)

print(dictionary.token2id)

#%%

from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api

# Download dataset
dataset = api.load("text8")
data = [d for d in dataset]

# Split the data into 2 parts. Part 2 will be used later to update the model
data_part1 = data[:1000]
data_part2 = data[1000:]

# Train Word2Vec model. Defaults result vector size = 100
model = Word2Vec(data_part1, min_count = 0, workers=cpu_count())

# Get the word vector for given word
topic_vec = model['topic']
#> array([ 0.0512,  0.2555,  0.9393, ... ,-0.5669,  0.6737], dtype=float32)

model.most_similar('topic')
#> [('discussion', 0.7590423822402954),
#>  ('consensus', 0.7253159284591675),
#>  ('discussions', 0.7252693176269531),
#>  ('interpretation', 0.7196053266525269),
#>  ('viewpoint', 0.7053568959236145),
#>  ('speculation', 0.7021505832672119),
#>  ('discourse', 0.7001898884773254),
#>  ('opinions', 0.6993060111999512),
#>  ('focus', 0.6959210634231567),
#>  ('scholarly', 0.6884037256240845)]

# Save and Load Model
model.save('newmodel')
model = Word2Vec.load('newmodel')

#%%

# Update the model with new data.
model.build_vocab(data_part2, update=True)
model.train(data_part2, total_examples=model.corpus_count, epochs=model.iter)
model['topic']
# array([-0.6482, -0.5468,  1.0688,  0.82  , ... , -0.8411,  0.3974], dtype=float32)

model.most_similar('topic')


#%% 


# Doc2vec

import gensim
import gensim.downloader as api

# Download dataset
dataset = api.load("text8")
data = [d for d in dataset]


# Create the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

train_data = list(create_tagged_document(data))

print(train_data[:1])
#> [TaggedDocument(words=['anarchism', 'originated', ... 'social', 'or', 'emotional'], tags=[0])]



#%%

# Init the Doc2Vec model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# Build the Volabulary
model.build_vocab(train_data)

# Train the Doc2Vec model
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

#%%

# Google word2vec


import gensim
import gensim.downloader as api
path = api.load("word2vec-google-news-300", return_path=True)
print(path)


from gensim.models.keyedvectors import KeyedVectors
gensim_model = KeyedVectors.load_word2vec_format(path, binary=True, limit=300000)

gensim_model['regression']

gensim_model.most_similar(positive=['statistics', 'diagnostics', 'outlier'])

gensim_model.most_similar(positive=['Tea', 'United_States'], negative=['England'])

gensim_model.most_similar(positive=['statistics', 'diagnostics', 'outlier'])

#%%


# Wikipedia2vec - Studio Ousia

# from wikipedia2vec import Wikipedia2Vec

# wiki2vec = Wikipedia2Vec.load(MODEL_FILE)


