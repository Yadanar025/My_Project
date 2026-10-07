

import os
import re
import csv
import pandas as pd
import preprocessing
import tfidf



def get_student_id():

    report_path = os.path.join(os.path.dirname(__file__), 'report.md')
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'-\s*\*\*MMDT ID:\*\*\s*(.*)', content)
                if match:
                    val = match.group(1).strip()
                    if val and not (val.startswith('[') and val.endswith(']')):
                        return re.sub(r'[^a-zA-Z0-9_-]', '_', val)
        except Exception:
            pass
    return "student"


def studentverifier(data_dir="data", output_dir="data", student_id=None):
    """
    Main entry point for Myanmar text claim verification and evidence retrieval.

    Pipeline Steps:
    1. Myanmar Text Preprocessing & Syllable/Word Tokenization (UTF-8)
    2. Feature Extraction (TF-IDF / Syllable N-grams / Dense Embeddings)
    3. Evidence Retrieval (Vector / Lexical similarity search from corpus.csv)
    4. Stance / Claim Classification (SUPPORTS / REFUTES / NOT ENOUGH INFO)
    5. Evidence-Grounded Explanation / RAG Synthesis in Myanmar context

    Args:
        data_dir (str): Directory containing input CSV files (corpus.csv, claims.csv, train_claims.csv, test_claims.csv).
        output_dir (str): Directory where '<studentid>_predictions.csv' will be saved.
        student_id (str, optional): The student ID to use in the filename. Defaults to ID in report.md.

    Returns:
        str: Path to the generated '<studentid>_predictions.csv' file.
    """
    if student_id is None:
        student_id = get_student_id()

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{student_id}_predictions.csv"
    output_path = os.path.join(output_dir, output_filename)

    # =========================================================================
    #
    # 1. Load corpus.csv, train_claims.csv, and test_claims.csv dynamically with utf-8 encoding.
    top_k = 5
    claimed_path = os.path.join(data_dir,"train_claims.csv")
    corpus_path = os.path.join(data_dir,"corpus.csv")
    df_claimed = pd.read_csv(claimed_path)
    df_corpus = pd.read_csv(corpus_path)
    test_path = os.path.join(data_dir,"test_claims.csv")
    df_test = pd.read_csv(test_path)
    # 2. Preprocess Myanmar text (Unicode normalization, syllable segmentation / regex tokenization, stopword removal).
    # 3. Build TF-IDF / Vector space representations for Myanmar documents and claims.
    y_test, y_pred, model, vectorizer, corpus_vectorizer, corpus_vector, df_corpus, df_claimed = tfidf.model(df_claimed,df_corpus)
   
    # 4. Perform Top-K evidence retrieval for each claim against corpus passages.
    # 5. Classify claims using Naïve Bayes baseline and/or RAG verification.
    result = []
    for _,row in df_test.iterrows():
        claim_id = row["claim_id"]
        claim_text= row["claim_text"]
        evidence = tfidf.retrieve_doc(claim_text, corpus_vectorizer, corpus_vector, df_corpus, top_k=top_k, threshold=0.1)
        if not evidence:
            predicted_label = "NOT ENOUGH INFO"
            retrieved_evidence_id= None
            confidence = 0.0
            evidence_text = "လုံလောက်သော အလားတူအထောက်အထား မတွေ့ရှိပါ။"
        else:
            top_evidence = evidence[0]
            claim_vector = vectorizer.transform([preprocessing.preprocess_text(claim_text,as_string=True)])
            predicted_label = model.predict(claim_vector)[0]
            confidence = top_evidence["score"]
            retrieved_evidence_id = top_evidence["doc_id"]
            evidence_text = top_evidence["document"]
        result.append({ "claim_id": claim_id,
            "predicted_label": predicted_label,
            "retrieved_evidence_id": retrieved_evidence_id,
            "confidence": confidence,
            "explanation": f"အထောက်အထား {predicted_label} အပေါ် အခြေခံ၍ {retrieved_evidence_id}:\"{evidence_text}\"ကို ခန့်မှန်းထားသည်"})
        
    # 6. Save predictions to `output_path`.
    #
    # Expected CSV columns:
    # claim_id,predicted_label,retrieved_evidence_id,confidence,explanation
    #
    # Labels must be one of: 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'
    # =========================================================================
    pd.DataFrame(result).to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    generated_file = studentverifier()
    print(f"Myanmar claim verification completed. Output file: {generated_file}")
