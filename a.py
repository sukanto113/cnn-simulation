#!/usr/bin/env python
# coding: utf-8

# In[7]:


def a_func():
    print("I am a func")

if(__name__=="__main__"):
    get_ipython().system('jupyter nbconvert a.ipynb --to python')
    a_func()

