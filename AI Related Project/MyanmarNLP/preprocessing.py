import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import pyidaungsu as pds

def stopword(path):
    with open(path,encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

stopwords = stopword("stopwords_word2vec_frequency.txt")| stopword("stopwords_fasttext_frequency.txt") 

def preprocess_text(text:str,remove_stopwords:bool=True,as_string:bool=False):
    """Preprocessing 
    1. Lowercases, 
    2. Strips punctuation, 
    3. Tokenizes, 
    4. Remove Stop words
    and retun list of tokens or strings"""  

    if not isinstance(text,str):
        text = str(text) if text is not None else ""

    #lower case
    text=text.lower()

    #strip punctuation
    text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text)
    text = re.sub(r'[^\u1000-\u109F\s.]', ' ', text)

    #Tokenize
    tokens = pds.tokenize(text,form ="word")

    #remove stopwords
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    #others
    if as_string:
        return " ".join(tokens)

    return tokens

if __name__=="__main__":
    df = pd.read_csv("./data/corpus.csv")
  
    print("Defult")
    print(df["text"].apply(lambda x:preprocess_text(x, as_string=True)))
