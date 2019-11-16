#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
import seaborn as sns
import numpy as np
from numpy.random import randn
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr
from collections import Counter


# In[75]:


df = pd.read_csv("Desktop/review_data.csv")


# In[76]:


#columns = ['title_x', 'url', 'image', 'reviewUrl', 'prices', 'name', 'verified', 'helpfulVotes']
#df = pd.DataFrame(sentiment.drop(columns,axis=1,inplace=False))


# In[77]:


df.head()


# In[78]:


df['rating'].value_counts().plot(kind='bar')


# In[79]:


## Change the reviews type to string
df['body'] = df['body'].astype(str)


# In[80]:


## Before lowercasing 
df['body'][2]


# In[81]:


## Lowercase all reviews
df['body'] = df['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['body'][2]


# In[82]:


## remove punctuation
df['body'] = df['body'].str.replace('\W',' ')
df['body'][2]


# In[83]:


stop = stopwords.words('english')
df['body'] = df['body'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['body'][2]


# In[84]:


st = PorterStemmer()
df['body'] = df['body'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df['body'][2]


# In[85]:


## Define a function which can be applied to calculate the score for the whole dataset

def getPol(x):
    return TextBlob(x).sentiment.polarity

def getSub(x):
    return TextBlob(x).sentiment.subjectivity

def getSent(x):
    if x == 0:
      return 'The text is neutral'
    elif x > 0:
      return 'The text is positive'
    else:
      return 'The text is negative'
    

df['senti_score_polarity'] = df['body'].apply(getPol)
df['senti_score_subjectivity'] = df['body'].apply(getSub)
df['sentiment'] = df['senti_score_polarity'].apply(getSent)
df.head()


# In[53]:


df.describe()


# In[54]:


#boxplot for df

boxplot = df.boxplot(column=['senti_score_polarity','senti_score_subjectivity'], 
                     fontsize = 15,grid = True, vert=True,figsize=(7,7,))
plt.ylabel('Range')


# In[55]:


#scatter for df

sns.lmplot(x='senti_score_subjectivity',y='senti_score_polarity',data=df,fit_reg=True,scatter=True, height=7,palette="mute") 


# In[56]:


# prepare data
data1 = df['senti_score_subjectivity']
data2 = data1 + df['senti_score_polarity']
# calculate covariance matrix
covariance = cov(data1, data2) 
print(covariance)

corr, _ = pearsonr(data1, data2)
print('correlation: %.5f' % corr)


# In[57]:


#Polarity Distribution for dffilter

plt.hist(df['senti_score_polarity'], color = 'darkred', edgecolor = 'black', density=False,
         bins = int(30))
plt.title('Polarity Distribution')
plt.xlabel("Polarity")
plt.ylabel("Number of TImes")

from pylab import rcParams
rcParams['figure.figsize'] = 10,15


# In[61]:


sns.distplot(df['senti_score_polarity'], hist=True, kde=True, 
             bins=int(30), color = 'darkred',
             hist_kws={'edgecolor':'black'},axlabel ='Polarity')
plt.title('Polarity Density')

from pylab import rcParams
rcParams['figure.figsize'] = 3,3


# In[91]:


words = df['body']

# generate DF out of Counter
rslt = pd.DataFrame(Counter(words).most_common(10),
                    columns=['Word', 'Frequency']).set_index('Word')
rslt


# In[92]:


explode= (0.1, 0.12, 0.122, 0,0,0,0,0,0,0)
labels=['good',
       'love',
       'great phone',
       'excel',
       'great',
       'love phone',
       'good phone',
       'perfect',
       'excelent',
       'work great']

plt.pie(rslt['Frequency'], explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=False, startangle=90)
plt.legend(labels, loc='lower left', fontsize='x-small',markerfirst=True)
plt.tight_layout()
plt.title('Common Words')
plt.show


# In[ ]:




