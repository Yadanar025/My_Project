
## Section 1: Dataset & Corpus Configuration
- **Total Corpus Documents Configured:** 28
- **Total Labeled Claims Configured:** 40
- **Total Test Claims Configured:** 10
- **Total Claims in Dataset:** 50

---


## Section 2: Project Structure
- **Project Directory Structure:** There are three main file 
-- preprocessing.py where i did data cleaning, tokenization and stopword removal. For the tokenization i use pyidaungsu.
-- tfidf.py where the core pipeline of TFIDF+Naive Bayes training and performance result is produce
-- app.py in which it connect every thing together and produce the prediction file
When the app.py's studentverifier() loads the data file through the tfidf.model() which fits a TfidfVectorizer on the claim text for classification and a separate TfidfVectorizer on corpus text for retrieval. While each row of test_claims.csv is found using retrieve_doc() that look for the most similar corpus passage using cosine similarity and  the claim is predict using the trained NB model and the MMDT002_predictions.csv is produced.
-- stopswords_tasttext_frequency.txt, stopswords_fasttext_frequency.txt is the file where i get the stopwords from bumese
