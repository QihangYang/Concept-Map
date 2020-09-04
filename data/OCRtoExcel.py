# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:10:38 2020

@author: Qihang Yang
"""



import os
import xlwt
from PIL import Image
import pytesseract as pt

#%%

workbook = xlwt.Workbook(encoding='utf-8')       
sheet = workbook.add_sheet("ALRM")         # Name
sheet.write(0, 0, "Problem Set ID")      
sheet.write(0, 1, "Problem text")     
sheet.write(0, 2, "Related Concept")  


path = r"C:\Users\Qihang Yang\Desktop\EducationalUI\newData\Jiarui\screenshots\lr_ALRM" # Folder location 
files= os.listdir(path) 

for n in range(len(files)):   
    file = files[n]
    img = Image.open(path + "\\" + file)
    ques = pt.image_to_string(img)
    sheet.write(n+1, 0, file[:-4])
    sheet.write(n+1, 1, ques)
    
    
workbook.save(r'C:\Users\Qihang Yang\Desktop\EducationalUI\newData\lr_ALRM.xls')  # Save excel
		

#%%

img1 = Image.open("images/image1.png")
ques1 = pt.image_to_string(img1)
print(ques1)
