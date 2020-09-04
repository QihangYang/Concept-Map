# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:40:59 2020

@author: Qihang Yang
"""

# Reference: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

#%% Packages

import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from random import sample
from math import floor

from gensim.models.word2vec import Word2Vec

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

#%% Functions

def If_Concept(problems_df, concept):  
    concepts = []
    for n in problems_df.index:
        if type(problems_df["Related_Concept"][n]) != str:
            concepts.append(0)
            continue
        if fuzz.partial_ratio(concept, problems_df["Related_Concept"][n]) >= 90:
            concepts.append(1)
        else:
            concepts.append(0)
    return np.array(concepts)

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def Text_To_Embedding(problem_text, embed_model, max_width = 500):
    embed_list = []
    words_list = problem_text.split()
    for n in range(len(words_list)):
        if words_list[n] in stopwords or len(words_list[n]) < 2:
            continue
        else:
             try:
                 embed_list.append(embed_model[words_list[n]])
             except:
                 pass
       
    if len(embed_list) < max_width:    
        for n in range(len(embed_list), max_width):
            embed_list.append(np.zeros(embed_model.vector_size))
    else:
        embed_list = embed_list[:max_width]
    return np.array(embed_list)

    
def Create_Embedding(problems, embed_model, max_width = 500):
    embed_mat = []
    for index in problems.index:
        if type(problems[index]) != str:
            embed_mat.append(Text_To_Embedding(" ", embed_model, max_width = max_width))
            continue
        embed_mat.append(Text_To_Embedding(problems[index], embed_model, max_width = max_width))
    return np.array(embed_mat)


#%% Data

problems = pd.read_csv("problem_2.csv")
embed_model = Word2Vec.load("embed_model_2")

# Choose a concept
concept_text = "Simple Linear Regression" # 226/295
#concept_text = "Multiple Linear Regression" # 152/369
#concept_text = "Hypothesis Testing" #112/409
#concept_text = "Linear Algebra" #91/430
#concept_text = "Estimation" # 183/338

# Training and test sets
Full_concept = If_Concept(problems, concept_text)
Full_pos = problems[Full_concept == 1]
print("Positive cases: %i" % Full_pos.shape[0])
Full_neg = problems[Full_concept == 0]
print("Negative cases: %i" % Full_neg.shape[0])

np.random.seed(1)
Train_pos = Full_pos.sample(floor(0.9 * Full_pos.shape[0]))
Test_pos = Full_pos.drop(Train_pos.index)
Train_neg = Full_neg.sample(floor(0.9 * Full_neg.shape[0]))
Test_neg = Full_neg.drop(Train_neg.index)

Train_problems_df = Train_pos.append(Train_neg)
Train_problems = Train_problems_df["Problem_text"]
Test_problems_df = Test_pos.append(Test_neg)
Test_problems = Test_problems_df["Problem_text"]

Train_X = Create_Embedding(Train_problems, embed_model, max_width = 50)
Test_X = Create_Embedding(Test_problems, embed_model, max_width = 50)

Train_Y = If_Concept(Train_problems_df, concept_text)
Test_Y = If_Concept(Test_problems_df, concept_text)


#%% Sequence Classification


# create the model
model = Sequential()
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(50))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) # sigmoid
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(Train_X, Train_Y, validation_data=(Test_X, Test_Y), epochs=10, batch_size=64)

print(model.summary())

#%% Test Evaluation

scores = model.evaluate(Test_X, Test_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100)) 

#%% Prediction

prediction = np.transpose([a for a in model.predict_classes(Test_X)])
prediction
Test_Y

#%% Results

# Simple Linear Regression 73.58%
Simple_prediction = prediction
Simple_Test_y = Test_Y
Simple_prediction
Simple_Test_y



#%% Show Output

output = pd.DataFrame(
    data = {"Problem Set ID": Test_problems_df["Problem Set ID"],
            "Problem text": Test_problems,
            "Concept": Test_Y,
            "Prediction": [a[0] for a in model.predict_classes(Test_X)]},
    columns = ["Problem text", "Concept", "Prediction"]
    )
print(output)

#%%

output.to_csv(r'C:\Users\Qihang Yang\Desktop\EducationalUI\word2vec\output.csv', index = False)

