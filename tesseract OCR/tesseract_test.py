# Tesseract test

from PIL import Image
import pytesseract as pt
import numpy as np
import pandas as pd

#%%
# Import image
img1 = Image.open("images/image1.png")
img1
ques1 = pt.image_to_string(img1)
print(ques1)


img2 = Image.open("images/image2.png")
img2
ques2 = pt.image_to_string(img2)
print(ques2)


Concepts_data = {"concepts" : ["random variable"],
                 "meaning" : ["a variable whose values depend on outcomes of a random phenomenon"],
                 "other" : [""]}

Concepts_database = pd.DataFrame(Concepts_data)
#print(Concepts_database)

#%%

# One hot encoding

# Initial concepts list
Concepts_list = ["quesID",
                 "random variable", 
                 "conditional probability", 
                 "bayes' theorem", 
                 "normal distribution"]

Questions_concepts_mat = pd.DataFrame(columns = Concepts_list)
Questions_concepts_mat



# Match question with concepts list
def match(ques, Concepts_list):
    tags = []
    for i in range(len(Concepts_list)):
        if ques.find(Concepts_list[i]) > 0:
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
    
    return(Questions_concepts_mat)


# Question 1
Questions_concepts_mat = add_question(quesID = "0001",
                                        ques = ques1, 
                                        Questions_concepts_mat = Questions_concepts_mat)
Questions_concepts_mat

# Question 2
Questions_concepts_mat = add_question(quesID = "0002",
                                        ques = ques2, 
                                        Questions_concepts_mat = Questions_concepts_mat)
Questions_concepts_mat


#%%

# Add concept function
def add_concept(con, Questions_concepts_mat):
    if con in np.array(Questions_concepts_mat.columns):
        print("The concept is already included.")
    else:      
        Questions_concepts_mat[con] = np.zeros(len(Questions_concepts_mat))
        print("The concept is added.")
    return(Questions_concepts_mat)



Questions_concepts_mat = add_concept(con = "probability",
                                     Questions_concepts_mat = Questions_concepts_mat)
Questions_concepts_mat


# Question 1
Questions_concepts_mat = add_question(quesID = "0001",
                                        ques = ques1, 
                                        Questions_concepts_mat = Questions_concepts_mat)
Questions_concepts_mat

# Question 2
Questions_concepts_mat = add_question(quesID = "0002",
                                        ques = ques2, 
                                        Questions_concepts_mat = Questions_concepts_mat)
Questions_concepts_mat









