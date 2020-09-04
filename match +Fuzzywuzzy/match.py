# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:08:57 2020

@author: Qihang Yang
"""


from PIL import Image
import pytesseract as pt
import numpy as np
import pandas as pd
import re

#%%
img1 = Image.open("linear1.png")
page1 = pt.image_to_string(img1, lang = "eng+equ")
print(page1)


#%%

page1_str = page1.split("\n\n")
print(page1_str)

#%%

for str in page1_str:
    if len(str) < 15:
        page1_str.remove(str)
    
for str in page1_str:
    if len(str) < 15:
        page1_str.remove(str)
        
    
print(page1_str)


#%%

file = open("concepts.txt")
Concepts_list = file.readlines()
Concepts_list = Concepts_list[0].split(",")
Concepts_list.pop()
Concepts_list.insert(0, "quesID")
print(Concepts_list)

#%%

Questions_concepts_mat = pd.DataFrame(columns = Concepts_list)
print(Questions_concepts_mat)

#%%

# Match question with concepts list
def match(ques, Concepts_list):
    tags = []
    for i in range(len(Concepts_list)):
        if bool(re.search(Concepts_list[i], ques, re.IGNORECASE)):
            tags.append(Concepts_list[i])
    return(tags)

# Add question to the questions-concepts matrix
def add_question(quesID, ques, Questions_concepts_mat):
    Concepts_list = Questions_concepts_mat.columns[1:]
    tags = match(ques, Concepts_list)
    ques_concepts = np.zeros(len(Concepts_list))
    for i in range(len(tags)):
        ques_concepts = np.where(Concepts_list == tags[i], 1, ques_concepts)
    
    if quesID in np.array(Questions_concepts_mat["quesID"]):
        if (Questions_concepts_mat.loc[Questions_concepts_mat["quesID"] == quesID, 
                                      Questions_concepts_mat.columns != "quesID"] == ques_concepts).values.all():
            print("The question is already included.")
        else:
            Questions_concepts_mat.loc[Questions_concepts_mat["quesID"] == quesID, 
                                      Questions_concepts_mat.columns != "quesID"] = ques_concepts
            print("The question is updated.")
            
    else:
        Questions_concepts_mat.loc[len(Questions_concepts_mat), "quesID"] = quesID
        Questions_concepts_mat.loc[Questions_concepts_mat["quesID"] == quesID, 
                                      Questions_concepts_mat.columns != "quesID"] = ques_concepts
        print("The question is added.")
    
    
    if sum(ques_concepts) == 0:
        print("No concept is matched")
    
    
    return(Questions_concepts_mat)

#%%
   
img1 = Image.open("images/ques1.png")
ques1 = pt.image_to_string(img1, lang = "eng+equ")

img2 = Image.open("images/ques2.png")
ques2 = pt.image_to_string(img2, lang = "eng+equ")
 
img3 = Image.open("images/ques3.png")
ques3 = pt.image_to_string(img3, lang = "eng+equ")
 
img4 = Image.open("images/ques4.png")
ques4 = pt.image_to_string(img4, lang = "eng+equ")
 
img5 = Image.open("images/ques5.png")
ques5 = pt.image_to_string(img5, lang = "eng+equ")
 
img6 = Image.open("images/ques6.png")
ques6 = pt.image_to_string(img6, lang = "eng+equ")
 
img7 = Image.open("images/ques7.png")
ques7 = pt.image_to_string(img7, lang = "eng+equ")
 
img8 = Image.open("images/ques8.png")
ques8 = pt.image_to_string(img8, lang = "eng+equ")
 
img9 = Image.open("images/ques9.png")
ques9 = pt.image_to_string(img9, lang = "eng+equ")
 
img10 = Image.open("images/ques10.png")
ques10 = pt.image_to_string(img10, lang = "eng+equ")
 


#%%

Questions_concepts_mat = add_question(quesID = "0001",
                                        ques = ques1, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0002",
                                        ques = ques2, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0003",
                                        ques = ques3, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0004",
                                        ques = ques4, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0005",
                                        ques = ques5, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0006",
                                        ques = ques6, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0007",
                                        ques = ques7, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0008",
                                        ques = ques8, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0009",
                                        ques = ques9, 
                                        Questions_concepts_mat = Questions_concepts_mat)

Questions_concepts_mat = add_question(quesID = "0010",
                                        ques = ques10, 
                                        Questions_concepts_mat = Questions_concepts_mat)


#%%

print(Questions_concepts_mat)


#%%


# FuzzyWuzzy

# pip install fuzzywuzzy
# pip install python-Levenshtein


from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#%%

fuzz.ratio("this is a test", "this is a test!")
fuzz.partial_ratio("linear regression", "llnear regression")
fuzz.partial_ratio("linear regression", "ilnear regression")
fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
fuzz.token_set_ratio("linear regression", "ilnear regression")


fuzz.partial_ratio("linear regression", "ilnear regresions normal distribution probability mathematics 123456789")

fuzz.partial_ratio("linear regression", "LINEAR regression")


choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
process.extract("new york jets", choices, limit=2)
process.extractOne("cowboys", choices)

