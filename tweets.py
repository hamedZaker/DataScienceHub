import pandas as pd
import numpy as np
import scipy as sc
import category_encoders as ce
import multiprocessing as mp
from joblib import Parallel, delayed

## NLP library
import re
import string
import nltk
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
nltk.download('wordnet')
from spellchecker import SpellChecker


## ML Library
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# import pickle

## Visualization library
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
# from wordcloud import WordCloud

## Ignoring Warning during trainings 
import warnings
warnings.filterwarnings('ignore')

## spell-checking text
def correctText(text):
    spell = SpellChecker(distance=1)
    text = text.split()
    text = Parallel(n_jobs=2)(delayed(spell.correction)(word) for word in text)
    return ' '.join(text)


## finding unique words
def uniqueWords(df):
    unique_words = {}
    for text in df.text:
        for word in text.split():
            if (word not in unique_words.keys()):
                unique_words[word] = 1
            else:
               unique_words[word] += 1
    return unique_words

## removing words in refList from text
def removeWords(text):
        text = text.split()
        for word in refList:
            try:
                text.remove(word)
            except ValueError:
                pass
        return ' '.join(text)
            

## text cleaning (lower case, unwanted characters, etc)
def textPrepare(df):        
    df.text = df.text.apply(lambda x:x.lower() ) # lowering the case
    df.text = df.text.apply(lambda x:re.sub('\[.*?\]', '', x) ) # remove square brackets
    df.text = df.text.apply(lambda x:re.sub('<.*?>+', '', x) )
    df.text = df.text.apply(lambda x:re.sub('https?://\S+|www\.\S+', '', x) ) # remove hyperlink
    df.text = df.text.apply(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) ) # remove puncuation
    df.text = df.text.apply(lambda x:re.sub('\s+', ' ', x) ) # remove leading/trailing/and extra spaces
    df.text = df.text.apply(lambda x:re.sub('\n' , '', x) ) # remove line breaks
    df.text = df.text.apply(lambda x:re.sub('\w*\d\w*' , '', x) ) # remove words containing numbers
    
    # spell cheking text
    # with mp.Pool(mp.cpu_count()) as pool:
        # df.text = pool.map(correctText, df.text)
    
    token = nltk.tokenize.RegexpTokenizer(r'\w+')     # Tokenizer
    df.text = df.text.apply(lambda x:token.tokenize(x))   # applying token
    df.text = df.text.apply(lambda x:[w for w in x if w not in cachedStopWords]) # removing stopwords

    stemmer = nltk.stem.PorterStemmer() # stemmering the text and joining
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df.text = df.text.apply(lambda x:" ".join(lemmatizer.lemmatize(token) for token in x))
    return df


if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # clean data
    train = textPrepare(train)
    test = textPrepare(test)

    unique_words = pd.DataFrame.from_dict(uniqueWords(train), orient='index', columns=['frequency'])
    unique_words.sort_values(by=["frequency"], inplace=True, ascending=False)
    refList = unique_words[unique_words['frequency'] < 5].index
    
    with mp.Pool(mp.cpu_count()) as pool:
        train.text = pool.map(removeWords, train.text)
        test.text = pool.map(removeWords, test.text)        

    print("----- train cleaned -----\n", train.head())
    
    count_vectorizer = CountVectorizer()
    train_vectors_count = count_vectorizer.fit_transform(train['text'])
    print(train_vectors_count.shape)
    test_vectors_count = count_vectorizer.transform(test["text"])

    X_train = train_vectors_count
    y_train = train["target"]
    X_test = test_vectors_count

    # Dummy classifier
    from sklearn.dummy import DummyClassifier
    dummy_model = DummyClassifier(strategy="uniform")
    scores = cross_val_score(dummy_model, X_train, y_train, cv=6, scoring="accuracy")
    print("Dummy accuracy: %.3f" % scores.mean())

    # Logistic Regression classifier
    from sklearn.linear_model import LogisticRegression
    reg_model = LogisticRegression(C=2, max_iter=300)
    scores = cross_val_score(reg_model, X_train, y_train, cv=6, scoring="accuracy")
    print("Logistic regression accuracy: %.3f" % scores.mean())


    # Naive Bayes classifer
    from sklearn.naive_bayes import MultinomialNB
    nb_model = MultinomialNB()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(nb_model, X_train, y_train, cv=cv, scoring="accuracy")
    print("Naive Bayes accuracy: %.3f" % scores.mean())

    # support vector classifier
    from sklearn.svm import SVC
    svm_model = SVC()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(svm_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print("SVM accuracy: %.3f" % scores.mean())


    # random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print("RandomForest accuracy: %.3f" % scores.mean())

    # voting
    from sklearn.ensemble import VotingClassifier
    voting_model = VotingClassifier([('reg',reg_model), ('bayes',nb_model), 
                                     ('svm',svm_model)], n_jobs=-1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(voting_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print("voting accuracy: %.3f" % scores.mean())

    print(pd.DataFrame(confusion_matrix(y_train, voting_model.fit(X_train, y_train).predict(X_train))))


    final_model = voting_model
    sample_submission = pd.read_csv("sample_submission.csv")
    pred = final_model.fit(X_train, y_train).predict(X_test)
    sample_submission["target"] = pred
    sample_submission.to_csv("submission.csv", index=False)








