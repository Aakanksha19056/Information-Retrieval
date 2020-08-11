#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import datasets
import csv
import itertools
import sys
import nltk
import glob
import os
import string


# In[3]:


g_paths = []
for (dirpath, dirnames, filenames) in os.walk(str(os.getcwd())+'/'+"comp.graphics"+'/'):
    for i in filenames:
        g_paths.append(str(dirpath)+i)
   


# In[4]:


r_paths = []
for (dirpath, dirnames, filenames) in os.walk(str(os.getcwd())+'/'+"rec.motorcycles"+'/'):
    for i in filenames:
        r_paths.append(str(dirpath)+i)
   


# In[5]:


final_paths = g_paths+r_paths


# In[6]:


print(final_paths)
print(len(final_paths))


# In[7]:


def open_files(file):
    with open(file, 'r') as file:
        data_orig_func = file.read()    
    #print(data_orig_func)
    #print(type(data_orig_func))
    return data_orig_func


#data_orig = open_files(final_paths[1])
#print(data_orig)
#print(len(data_orig))


# In[9]:


def conv_lowercase(data):
    data_lower_func = data.lower()
    #print(data_lower_func)
    return data_lower_func

#data_lower = conv_lowercase(data_orig)
#print(data_lower)
#print(len(data_lower))


# In[10]:


def rem_metadata(data):
    pos=data.index('\n\n')
    data_meta_func=data[pos:]
    return data_meta_func
    
#out_meta = rem_metadata(data_lower)
#print(out_meta)
#print(type(out_meta))


# In[11]:


def rem_punctuation(data):
    punc_marks=["!","@","#","$","%","^","&","*","(",")","-","=","\\","/","<",">","\\n","?","+","~","`"]
    for i in data:
        for j in range(len(punc_marks)):
            if(i==punc_marks[j]):
                data = np.char.replace(data,i," ")
    data = np.char.replace(data,","," ")
    data = np.char.replace(data, ":", " ")
    data = np.char.replace(data, "\"", " ")
    data = np.char.replace(data, "\"", " ")
    data = np.char.replace(data, ".", " ")
    data = np.char.replace(data, "\'", " ")
    data = np.char.replace(data, ";", " ")
    data = np.char.replace(data, "\\n", " ")
    data = np.char.replace(data, "|", " ")
    data = np.char.replace(data, "[", " ")
    data = np.char.replace(data, "]", " ")
    data = np.char.replace(data, "{", " ")
    data = np.char.replace(data, "}", " ")
    print("Output starts from here**********************")
    data_list =  data.tolist()
    return data_list
    
                
                
#out_rem =rem_punctuation(out_meta)
#print(out_rem)
#print(type(out_rem))

#x_list=x.tolist()
#print(x_list)
#print(len(x_list))


# In[12]:


from nltk.tokenize import word_tokenize
def token_gen(data):
    data_token = word_tokenize(data)
    return data_token

#out_token = token_gen(out_rem)
#print(out_token)
#print(type(out_token))


# In[13]:


keys = [item for item in range(1, 2000+1)]
doc_dict = dict(zip(final_paths, keys))
print(doc_dict)


# In[23]:


temp=[]
for i in range(0,100):
    temp.append(final_paths[i])
    
print(len(temp))


# In[17]:


pos_index={}
def generate_dict(data):
    filename = doc_dict.get(final_paths[1])
    for i in data:
        x = [j+1 for j,w in enumerate(data) if w.lower() == i]
        print(x)
        if i not in pos_index:
            pos_index[i]= {filename:x}
        else:
            pos_index[i][filename]=x
    print(pos_index)
generate_dict(out_token)


# In[27]:


pos_index={}
for i in final_paths:
    data_orig = open_files(i)
    data_lower = conv_lowercase(data_orig)
    out_meta = rem_metadata(data_lower)
    out_rem =rem_punctuation(out_meta)
    out_token = token_gen(out_rem)
    filename = doc_dict.get(i)
    for i in out_token:
        x = [j+1 for j,w in enumerate(out_token) if w.lower() == i]
        #print(x)
        if i not in pos_index:
            pos_index[i]= {filename:x}
        else:
            pos_index[i][filename]=x
    print(len(pos_index))
    


# In[28]:


print(len(pos_index))


# In[29]:


import itertools
def take(n,iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, 10))

n_items = take(10,pos_index.items())
print(n_items)


# In[62]:


def get_input():
    q=str(input())
    q_lower = conv_lowercase(q)
    q_rem = rem_punctuation(q_lower)
    q_token = token_gen(q_rem)
    print(q_token)
    k2=len(q_token)
    list_of_docs=[[] for i in range(k2)]
    for i in range(len(q_token)):
        x=pos_index.get(q_token[i])
        for key, value in x.items():
            list_of_docs[i].append(key)
    print(list_of_docs)
    actual_docs=[]
    if len(list_of_docs)>2:
        i=0
        j=i+1
        while i<len(list_of_docs) & j<len(list_of_docs):
            actual_docs=[x for x in list_of_docs[i] if x in list_of_docs[j]]
            i=i+1
            j=j+1
    else:
        actual_docs=[x for x in list_of_docs[0] if x in list_of_docs[1]]
    print(actual_docs)
    for i in range(len(actual_docs)):
        actual_docs[i]=str(actual_docs[i])
    print(actual_docs)
    z=[]
    #files=[]
    #import math
    def rep(number):
        return "{number:05}".format(number=int(number))
    for i in q_token:
        temp=[]
        try:
            #print("j = ",j)
            for j in (pos_index.get(i)):
                #print(x[i][j],"\n")
                for k in x[i][j]:
                    #print(k)
                    temp.append(rep(int(j))+k)
        except:
            m=1;
        z.append(temp)
    print(z,"\n")
    print(len(z))
    for i in range(8):
        try:
            print(list(list(zip(*z))[i]))
        except:
            op="im op"


get_input()
        
        


# In[61]:


l1=[1,2,3]
for i in range(len(l1)):
    l1[i]=str(l1[i])
print(l1)


# In[ ]:




