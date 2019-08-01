import pandas as pd
from helper import *
from sklearn.feature_extraction.text import TfidfVectorizer
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
import string

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 

def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    return two

df = pd.read_csv(open("abstracts for 'automat'.csv", errors='ignore'))
df=df.astype(str)


text = df.applymap(cleaning)['paperAbstract']
text_list = [i.split() for i in text]

import numpy as np

tfidf_vect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df=0.01)  

vector=tfidf_vect.fit_transform(text_list).todense()

vocab=tfidf_vect.vocabulary_ 

from sklearn import decomposition

num_top_words=2

def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]

d=10  #NUMBER OF TOPICS
clf = decomposition.NMF(n_components=d, random_state=1)

W1 = clf.fit_transform(vector)
H1 = clf.components_

#print (show_topics(H1))

for i,topic in enumerate(clf.components_):
    print('\n')
    print(f'Top 30 words for topic #{i+1}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-30:]])

topic_values = clf.transform(vector)

print (topic_values)