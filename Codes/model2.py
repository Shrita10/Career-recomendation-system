# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:33:12 2020

@author: Shrita
"""

import pandas as pd
import pickle
pd.options.display.max_columns = 500
df1 = pd.read_csv('C:\\Users\\Shrita\\Desktop\\Career-recomendation-system\\Codes\\job_rec.csv')
df1 = df1[df1['IT'] == True]
col = ['RequiredQual', 'Eligibility', 'Title', 'JobDescription', 'JobRequirment']
df1 = df1[col]

def repl(title):
    tokens = title.split()
    for i,x in enumerate(tokens):
        if x == 'Senior':
            tokens = tokens[1:]
        elif x == 'Junior':
            tokens = tokens[1:]
            
    title2 = ' '.join(tokens)
    return title2

df1['Title'] = df1['Title'].apply(lambda x: repl(x))


def repl(s):
    if 'C++ Software Developer' in s:
        s = s.replace('C++ Software Developer', 'C++ Developer')
    elif ('Quality Assurance Engineer' in s) or ('Software QA Engineer' in s):
        s = s.replace('Quality Assurance Engineer', 'QA Engineer')
        s = s.replace('Software QA Engineer', 'QA Engineer')
    elif '.Net Developer' in s:
        s = s.replace('.Net Developer', '.NET Developer')
    elif 'Java Software Developer' in s:
        s = s.replace('Java Software Developer', 'Java Developer')
    elif 'Senior Software Developer' in s:
        s = s.replace('Senior Software Developer', 'Software Developer')
    elif 'Database Administrator' in s:
        s = s.replace('Database Administrator', 'Database Developer')
    elif 'PHP Software Developer' in s:
        s = s.replace('PHP Software Developer', 'PHP Developer')
    elif '.NET Software Developer' in s:
        s = s.replace('.NET Software Developer', '.NET Developer')
    elif 'Java Software Engineer' in s:
        s = s.replace('Java Software Engineer', 'Java Developer')
    elif ('C#.NET Developer' in s) or ('C# .NET Developer' in s):
        s = s.replace('C#.NET Developer', '.NET Developer')
        s = s.replace('C# .NET Developer', '.NET Developer')
    elif 'ASP.NET Developer' in s:
        s = s.replace('ASP.NET Developer', 'Java Developer')
    
        
    else:
        pass
    
    return s



df1['Title'] = df1['Title'].apply(lambda x: repl(x))

classes = df1['Title'].value_counts()[:14]
keys = classes.keys().to_list()

df1 = df1[df1['Title'].isin(keys)]
df1['Title'].value_counts()

#dropping eligibility because more than 30% of the values are missing
df1.drop('Eligibility' , axis=1 , inplace =True)

from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()
df1['title_le'] = le2.fit_transform(df1['Title'])


pickle.dump(le2, open('le2.pkl','wb'))

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
        # lemmatize text - convert to base form 
        self.wnl = WordNetLemmatizer()
        # creating stopwords list, to ignore lemmatizing stopwords 
        self.stopwords = stopwords.words('english')
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.stopwords]

# removing new line characters, and certain hypen patterns                  
df1['RequiredQual']=df1['RequiredQual'].apply(lambda x: x.replace('\n', ' ').replace('\r', '').replace('- ', ''). replace(' - ', ' to '))


import nltk
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# train features and labels 
y = df1['title_le']
X = df1['RequiredQual']
# tdif feature rep 
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
vectorizer.fit(X)
# transoforming text to tdif features
tfidf_matrix = vectorizer.transform(X)
# sparse matrix to dense matrix for training
X_tdif = tfidf_matrix.toarray()

pickle.dump(vectorizer, open('vectorizer.pkl','wb'))


X_train_words, X_test_words, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)

X_train = vectorizer.transform(X_train_words)
X_train = X_train.toarray()

X_test = vectorizer.transform(X_test_words)
X_test = X_test.toarray()

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(C = 100, penalty = 'l2')

logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

pickle.dump(logistic, open('model2.pkl','wb'))

