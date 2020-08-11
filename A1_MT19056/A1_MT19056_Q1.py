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


# In[56]:


final_paths = []
for (dirpath, dirnames, filenames) in os.walk(str(os.getcwd())+'/'+"20_newsgroups"+'/'):
    for i in filenames:
        final_paths.append(str(dirpath)+str("/")+i)


# In[57]:


final_paths[1]


# In[58]:


len(final_paths)


# In[59]:


def open_files(file):
    with open(file, 'r') as file:
        data_orig_func = file.read()    
    #print(data_orig_func)
    #print(type(data_orig_func))
    return data_orig_func


#data_orig = open_files(paths[1])
#print(data_orig)
#print(len(data_orig))


# In[60]:


def conv_lowercase(data):
    data_lower_func = data.lower()
    #print(data_lower_func)
    return data_lower_func

#data_lower = conv_lowercase(data_orig)
#print(data_lower)
#print(len(data_lower))


# In[7]:


def rem_metadata(data):
    pos=data.index('\n\n')
    data_meta_func=data[pos:]
    return data_meta_func
    


# In[8]:


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
    
                
                
#x=rem_punctuation(data_lower)
#print(x)
#print(type(x))


# In[10]:


from nltk import word_tokenize


# In[219]:


import nltk
nltk.download('stopwords')


# In[11]:


from nltk.corpus import stopwords
sw_list = set(stopwords.words('english'))
print(sw_list)


# In[61]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def rem_stopwords(data):
    new_data=[]
    rmsw_list=[]
    sw_list = set(stopwords.words('english'))
    word_list = word_tokenize(data)
    #print(word_list)
    #print(type(word_list))
    for i in data.split():
        #print(i)
        for j in sw_list:
            if(i==j):
                #print(i)
                #print(j)
                new_data.append(i)
                new_data = list(set(new_data))
    rmsw_list = [ele for ele in word_list if ele not in new_data]
    return rmsw_list
    
    
#out_rmsw = rem_stopwords(x_list)
#print(out_rmsw)
#print(len(out_rmsw))
#print(type(out_rmsw))


# In[13]:


import nltk
nltk.download('punkt')


# In[14]:


nltk.download('wordnet')


# In[62]:


from nltk.stem import WordNetLemmatizer 

def lemm_data(data):
    list_lemm=[]
    for i in data:
        lemmatizer = WordNetLemmatizer()
        i = lemmatizer.lemmatize(i)
        list_lemm.append(i)
    return list_lemm

#out_lemm = lemm_data(out_rmsw)
#print(out_lemm)   
#print(len(out_lemm))


# In[14]:


import string
def rem_singleChar(data):
    char_list= list(string.ascii_lowercase)
    #print(char_list)
    new_list = [ele for ele in data if ele not in char_list]
    #print(len(new_list))
    return new_list
    
#rem_singleChar(out_lemm)


# In[15]:


temp=[]
for i in range(0,10):
    temp.append(paths[i])
    
#print(temp)
    
    


# **MAIN TOKEN DICTIONARY**

# In[16]:


keys = [item for item in range(1, 19997+1)]
doc_dict = dict(zip(paths, keys))
print(doc_dict)


# In[67]:


token_dict={}
for i in paths:
    data_orig = open_files(i)
    data_lower = conv_lowercase(data_orig)
    data_meta = rem_metadata(data_lower)
    x=rem_punctuation(data_meta)
    out_rmsw = rem_stopwords(x)
    out_lemm = lemm_data(out_rmsw)
    final_token_list = rem_singleChar(out_lemm)
    #print("without unique",len(final_token_list))
    #print(final_token_list)
    final_token_list = list(set(final_token_list))
    #print("wiht unique",final_token_list)
    #print(len(final_token_list))
    file_name = doc_dict.get(i)
    for i in final_token_list:
        if i not in token_dict:
            token_dict[i]= [file_name]
        else:
            token_dict[i].append(file_name)
print(len(token_dict))
        
            
            
        
    
    


# In[18]:


import itertools
def take(n,iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, 10))

n_items = take(10,token_dict.items())
print(n_items)


# In[78]:


for key in token_dict:
    l1=len(token_dict[key])
    token_dict[key].insert(0,l1)
    
n_items = take(10,token_dict.items())
print(n_items)


# In[34]:


def query_preprocessing(query):
    operands=[]
    operators=[]
    #print(query)
    #print(type(query))
    query_lower = conv_lowercase(query)    
    query_tokens = word_tokenize(query_lower)    
    #print(query_tokens)
    for i in query_tokens:
        if i not in ("and", "not", "or"):
            lemmatizer = WordNetLemmatizer()
            i = lemmatizer.lemmatize(i)
            operands.append(i)
        else:
            operators.append(i)
    #print(operands)
    #print(operators)
    return operators, operands

#print("Enter the query")
#q=str(input())
#query_preprocessing('hell ')

    
    


# In[29]:


def get_not_word_list(token_dict, doc_list,q):    
    word = 'not'
    not_operand_list=[]
    x = [i+2 for i,w in enumerate(q.split()) if w.lower() == word]
    not_word = q.split(' ')
    for i in x:
        not_operand_list.append(not_word[i-1])  
    print(not_operand_list)
    k2 = len(not_operand_list)
    not_pl=[[] for i in range(k2)]
    not_posting_list=[[] for i in range(k2)]
    for i in range(len(not_operand_list)):
        #print(not_operand_list[i])
        not_pl[i] = token_dict.get(not_operand_list[i])
    #print(not_pl)
    for i in range(len(not_pl)):
        not_pl[i].pop(0)
        l1 = [ele for ele in keys if ele not in not_pl[i]]
        not_posting_list[i] = l1
    for i in not_posting_list:
        i.insert(0,len(i))
        #print(len(i)) 
    #print(not_operand_list)
    #print(not_posting_list)
    return not_operand_list, not_posting_list
    
#get_not_word_list(token_dict, keys,'hell ')


# In[64]:


def merge_for_and(l1,l2):
    print("Inside merge")
    c=0
    l1_func = l1[1:]
    l2_func = l2[1:]
    #print(l1_func)
    #print(l2_func)
    res_and=[]
    #print(len(l1))
    n1=len(l1_func)
    n2=len(l2_func)
    #print(l1_func[0])
    #print(l2_func[0])
    i=0
    j=0
    while i<n1 and j<n2:
        if l1_func[i]==l2_func[j]:
            c=c+1
            res_and.append(l1_func[i])
            i=i+1
            j=j+1
        elif(l2_func[j]<l1_func[i]):
            c=c+1
            j=j+1
        else: 
            c=c+1
            i=i+1
    return res_and,c
    



# In[87]:


def get_input(token_dict, doc_list, doc_dict,q):
    #print("Enter the query")
    #q=str(input())
    print(q)
    operators, operands = query_preprocessing(q)
    not_op, not_post = get_not_word_list(token_dict, doc_list,q)
    #print(not_op)
    final_operands =  [ele for ele in operands if ele not in not_op]
    print(final_operands)
    k1 = len(final_operands)
    pl=[[] for i in range(k1)]   
    for i in range(len(final_operands)):
        #print(final_operands[i])
        pl[i] = token_dict.get(final_operands[i])
    #print(pl)
    #print(len(not_post))
    if(len(not_post)!=0):
        final_and_list = not_post + pl
        final_and_list.sort(key=lambda x: x[0])
    else:
        #print("Inside elif")
        final_and_list = pl
        final_and_list.sort(key=lambda x: x[0])
    i=0
    j=i+1
    final_comp = 0
    while i<len(final_and_list) and j<len(final_and_list):
        #print("Inside while")
        interim, comp_interim = merge_for_and(final_and_list[i],final_and_list[j])
        #interim1=list(set(final_and_list[i]) & set(final_and_list[j]))
        #print(interim)
        #print("Intersection anser", interim1)
        #print(type(interim))
        final_comp = final_comp + comp_interim
        final_and_list=final_and_list[j+1:]
        #print(f)
        final_and_list.insert(0,interim)
        i=i+1
        j=j+1
    #for i in final_and_list:
    #    i.insert(0,len(i))
    print("Documents retrieved",final_and_list)
    print("Number of documents retrieved",len(final_and_list[0]))
    print("Number of comaprisons made",final_comp)
    return final_and_list
          

#get_input(token_dict, keys, doc_dict,'hell')


# In[88]:


def get_input_or():
    q=str(input())
    or_ans=[]
    final_ans=[]
    q_lower = conv_lowercase(q)
    search_str = 'or'
    result = [i+1 for i,w in enumerate(q_lower.split()) if w.lower() == search_str]
    print(result)
    if (len(result)==0):
        get_input(token_dict, keys, doc_dict,q_lower)
    else:
        q_and = q_lower.split('or')
        print("After split of or",q_and)
        a=0
        for i in range(len(q_and)):
            print(i)
            inter = get_input(token_dict, keys, doc_dict,q_and[i])
            q_and[i]=inter
        c=0
        for i in q_and:
            c=c+1
        print(c)
        for i in q_and:
            for j in range(len(i)):
                or_ans.append(i[j])
        print(or_ans)
        for i in or_ans:
            final_ans = final_ans+i
        final_ans =  list(set(final_ans))
        print(final_ans)
        print(len(final_ans))

    
    
    
get_input_or()
    


# In[76]:


for key, value in doc_dict.items():
    if 86 == value:
        print(key)


# In[83]:


token_dict.get('maze')


# In[ ]:




