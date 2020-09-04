# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 00:14:26 2020

@author: Qihang Yang
"""

#%% 

from PIL import Image
import pytesseract as pt
import numpy as np
import pandas as pd

#%%
import pandas as pd

concepts_list = ["con1", "con2", "con3", "con4", "con5", "con6", "con7", "con8", "con9", "con10"]
ques_list = ["quesID1", "quesID2", "quesID3", "quesID4", "quesID5", "quesID6", "quesID7", "quesID8", "quesID9", "quesID10"]

np.random.seed(0)
ques_concepts = pd.DataFrame(data = np.random.binomial(size = 100, n = 1, p = 0.2).reshape(10, 10), 
                            index = ques_list, 
                            columns = concepts_list)
print(ques_concepts)

#%%

def distBtwQues(quesID1, quesID2, ques_concepts):
    ques1_con = ques_concepts.loc[quesID1]
    ques2_con = ques_concepts.loc[quesID2]
    if sum((ques1_con + ques2_con) != 0) != 0:
        Jdist = 1 - sum(ques1_con * ques2_con != 0) /sum((ques1_con + ques2_con) != 0)
    else:
        Jdist = 1
    return(Jdist)



#%%

def findSimQues(quesID, ques_concepts):
    ques_dis = []
    for quesID2 in ques_concepts.index[ques_concepts.index != quesID]:
        ques_dis.append(distBtwQues(quesID, quesID2, ques_concepts))
    sort_index = np.argsort(ques_dis)
    return(ques_concepts.index[sort_index])
  
#%%
    
# Questions distances

ques_dis_mat = pd.DataFrame(index = ques_list, columns = ques_list)
for quesID1 in ques_list:
    for quesID2 in ques_list:       
        ques_dis_mat.loc[quesID1, quesID2] = distBtwQues(quesID1, quesID2, ques_concepts)
print(ques_dis_mat)


#%% 

quesID1_min = findSimQues("quesID1", ques_concepts)
print(quesID1_min)


#%%

concepts_mat = pd.DataFrame(np.zeros(100).reshape(10, 10), index = concepts_list, columns = concepts_list)

concepts_mat.loc["con1", "con3"] = 1
concepts_mat.loc["con3", "con5"] = 1
concepts_mat.loc["con5", "con7"] = 1
concepts_mat.loc["con5", "con4"] = -1
concepts_mat.loc["con4", "con8"] = 1

print(concepts_mat)

#%%

concepts_graph = {"con1": ["con3", "con5", "con7", "con9"],
                  "con2": ["con4"],
                  "con3": ["con5"],
                  "con4": ["con6"],
                  "con5": ["con8"],
                  "con6": [],
                  "con7": ["con10"],
                  "con8": ["con10"],
                  "con9": [],
                  "con10": []}


#%%


def findShortestPath(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    
    shortestPath = []
    for node in graph[start]:
        if node not in path:
            newpath = findShortestPath(graph, node, end, path)
            if newpath:
                if not shortestPath or len(newpath)<len(shortestPath):
                    shortestPath = newpath
    return shortestPath


#%%

def findConPath (quesID, ques_concepts, concepts_mat):
    concepts = np.array(ques_concepts.loc[quesID, ques_concepts.loc[quesID] == 1].index)
    n_con = len(concepts)
    
    path_list = []
    for start in concepts:
        for end in concepts[concepts != start]:
            path = findShortestPath(concepts_graph, start, end, path_list)
            if path:
                path_list.append(path)
    return(path_list)




#%%

quesID3_path = findConPath("quesID3", ques_concepts, concepts_graph)
print(quesID3_path)   


#%%

def distToCons(input_cons, ques_concepts):
    cons_list = np.zeros(len(ques_concepts.columns))
    ques0 = pd.Series(cons_list, index = ques_concepts.columns)
    ques0[input_cons] = 1
    
    ques_dist = []
    for ques in np.array(ques_concepts.index):
        ques_con = ques_concepts.loc[ques]
        if sum((ques_con + ques0) != 0) != 0:
            Jdist = 1 - sum(ques_con * ques0 != 0) /sum((ques_con + ques0) != 0)
        else:
            Jdist = 1
        ques_dist.append(Jdist)
        
    sort_index = np.argsort(ques_dist)
    return(ques_concepts.index[sort_index])

#%%
    
input_cons = ["con1", "con8"]

recom_ques = distToCons(input_cons, ques_concepts)
print(recom_ques)