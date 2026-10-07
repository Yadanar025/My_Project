import preprocessing
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline


def model(df_claimed, df_corpus):


    label = df_claimed["label"]
    docs = df_claimed["claim_text"].apply(lambda x: preprocessing.preprocess_text(x, as_string=True))

    X_train, X_test, y_train, y_test = train_test_split(docs,label,test_size=0.2,random_state=42)

    cv_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2)),
    ("nb", MultinomialNB())])
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv_scores = cross_val_score(cv_pipeline, docs, label, cv=cv, scoring="f1_macro")

    vectorizer = TfidfVectorizer(min_df=min(2,len(X_train)))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    corpus_docs = df_corpus["text"].apply(lambda x: preprocessing.preprocess_text(x, as_string=True))

    corpus_vectorizer = TfidfVectorizer()
    corpus_vector = corpus_vectorizer.fit_transform(corpus_docs)

    return y_test, y_pred, model, vectorizer, corpus_vectorizer, corpus_vector, df_corpus, df_claimed


def retrieve_doc(claim, vectorizer, corpus_vectors, df, top_k=5, threshold=0.1):
    claim = preprocessing.preprocess_text(claim, as_string=True)

    claim_vector = vectorizer.transform([claim])

    similarities = cosine_similarity(claim_vector, corpus_vectors)[0]

    ranked = similarities.argsort()[::-1]

    max_score = similarities[ranked[0]]

    if max_score < threshold:
        return []

    results = []

    for index in ranked[:top_k]:
        results.append({"doc_id": df.iloc[index]["doc_id"],"document": df.iloc[index]["text"] , "score" : similarities[index]})

    return results

def performance(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average="macro",zero_division=0)
    recall = recall_score(y_test,y_pred, average="macro",zero_division=0)
    macro_f1 = f1_score(y_test,y_pred, average="macro",zero_division=0)
    matrix = confusion_matrix(y_test,y_pred)
    return accuracy,precision,recall,macro_f1,matrix

def retrieval_result(df_claimed, vectorizer, corpus_vectors, df_corpus, top_k=5, threshold=0.1):
    precision_score = []
    recall_score = []
    reciprocal_rank = []
    claims_with_evidence = df_claimed[df_claimed["evidence_id"].notna()]

    precision_scores_retrieve = []
    recall_scores_retrieve = []
    reciprocal_ranks_retrieve = []

    for _,row in claims_with_evidence.iterrows():
        claim = row["claim_text"]
        correct_doc_id = row["evidence_id"]
        results = retrieve_doc(claim,vectorizer,corpus_vectors,df_corpus,top_k=top_k,threshold=threshold)

        retrieved_ids = [result["doc_id"] for result in results]
        if correct_doc_id in retrieved_ids:
            precision_scores_retrieve.append(1 / len(retrieved_ids))
            recall_scores_retrieve.append(1)
            
            rank = retrieved_ids.index(correct_doc_id) + 1
            reciprocal_ranks_retrieve.append(1 / rank)

        else:
            precision_scores_retrieve.append(0)
            recall_scores_retrieve.append(0)
            reciprocal_ranks_retrieve.append(0)

    precision_at_k = sum(precision_scores_retrieve) / len(precision_scores_retrieve)
    recall_at_k = sum(recall_scores_retrieve) / len(recall_scores_retrieve)
    mrr = sum(reciprocal_ranks_retrieve) / len(reciprocal_ranks_retrieve)
    return precision_at_k, recall_at_k, mrr

if __name__ == "__main__":
    claimed_path = "./data/train_claims.csv"
    corpus_path = "./data/corpus.csv"
    top_k = 5
    df_claimed = pd.read_csv(claimed_path)
    df_corpus = pd.read_csv(corpus_path)
    y_test, y_pred, model, vectorizer, corpus_vectorizer, corpus_vectors, df_corpus, df_claimed = model(df_claimed,df_corpus)

    print(f"y test : {y_test.tolist()}, y pred : {y_pred.tolist()}")

    claim = "မြန်မာနိုင်ငံ၏ အကြီးဆုံးမြို့မှာ ရန်ကုန်မြို့ ဖြစ်သည်။"
    results = retrieve_doc(claim,corpus_vectorizer,corpus_vectors, df_corpus, top_k=5,threshold=0.1)
    accuracy,precision,recall,macro_f1,matrix = performance(y_test,y_pred)

    # print("CV Macro-F1 per fold:", cv_scores)
    # print("CV Macro-F1 mean:", cv_scores.mean())

    print("======================")

    print(results)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Macro-F1:", macro_f1)
    print("Confusion Matrix:")
    print(matrix)

    print("======================")

    precision_at_k,recall_at_k,mrr =retrieval_result(df_claimed, corpus_vectorizer, corpus_vectors,df_corpus, top_k=5,threshold=0.1)

    print("Precision@{}:".format(top_k), precision_at_k)
    print("Recall@{}:".format(top_k), recall_at_k)
    print("MRR:", mrr)

