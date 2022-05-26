import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from IPython.display import display

def get_text ():
    path = "./data/test-bio.csv"
    file = open(path)
    csvreader =  csv.reader(file)
    rows = []
    TextType = ' '
    for raw in csvreader:
        head, sep, tail = raw[0].partition('\t')
        if( '__END_' or ',' or '.' or '!' or '?' or ':' or ';'  not  in head):
            TextType = TextType+ ' ' +head
            rows.append(head.lower())
       # print(raw)

    print(len(rows))
  #  print(TextType)
    file.close()
    return rows
    
def bag_of_words(text):
    countVeector = CountVectorizer(min_df=0, max_df=1.)
    countVeectorMatrix = countVeector.fit_transform(text).toarray()

    words = countVeector.get_feature_names_out()
    pd.DataFrame(countVeectorMatrix, columns=words)

def tf_idf(TextArray):

    tfidf = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1, 2))
    features = tfidf.fit_transform(TextArray)
    features = features.toarray()
    print(features)

    vocab = tfidf.get_feature_names_out()
    
    data =  pd.DataFrame(np.round(features,2), columns=vocab)
    
    display(data)
  
   # pd.Dataframe{ features.todense(),columns = tfidf.get_feature_names() }

   
    
def main():
    ## YOUR CODE HERE
    print("it works!")
    tf_idf(get_text())
    #bag_of_words(get_text()) 
    pass


if __name__ == '__main__':
    main()