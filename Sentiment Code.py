#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob


# In[69]:


sentiment = pd.read_csv("Desktop/review_data.csv")


# In[70]:


columns = ['title_x', 'url', 'image', 'reviewUrl', 'prices', 'name', 'verified', 'helpfulVotes']


# In[71]:


df = pd.DataFrame(sentiment.drop(columns,axis=1,inplace=False))


# In[72]:


df['rating_x'].value_counts().plot(kind='bar')


# In[73]:


## Change the reviews type to string
df['body'] = df['body'].astype(str)

## Before lowercasing 
df['body'][2]

## Lowercase all reviews
df['body'] = df['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['body'][2]


# In[74]:


## remove punctuation
df['body'] = df['body'].str.replace('[^ws]','')
df['body'][2]


# In[56]:


stop = stopwords.words('english')
df['body'] = df['body'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['body'][2]


# In[57]:


st = PorterStemmer()
df['body'] = df['body'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df['body'][2]


# In[59]:


## Define a function which can be applied to calculate the score for the whole dataset

def senti(x):
    return TextBlob(x).sentiment  

df['senti_score'] = df['body'].apply(senti)

df.senti_score.head()


# In[ ]:




