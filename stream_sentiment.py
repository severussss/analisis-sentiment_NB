import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


def train_and_evaluate(
    df: pd.DataFrame,
    text_col: str = "Clean Tweet",
    label_col: str = "Score",
    test_size: float = 0.2,
    random_state: int = 42,
):
    if text_col not in df.columns:
        raise KeyError(f"Kolom teks '{text_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"Kolom label '{label_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    work = df[[text_col, label_col]].copy()
    work = work.dropna(subset=[text_col, label_col])
    work[text_col] = work[text_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        work[text_col],
        work[label_col],
        test_size=float(test_size),
        random_state=random_state,
        shuffle=True
    )

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(tfidf_train.toarray(), y_train)

    nb = MultinomialNB()
    nb.fit(X_train_res, y_train_res)

    y_pred = nb.predict(tfidf_test)
    cf_matrix = confusion_matrix(y_test, y_pred)

    report_text = classification_report(
        y_test, y_pred,
        zero_division=0
    )

    return {
        "model": nb,
        "vectorizer": tfidf_vectorizer,
        "cf_matrix": cf_matrix,
        "report_text": report_text,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_size": float(test_size),
    }