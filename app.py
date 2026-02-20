import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from textblob import TextBlob
from stream_sentiment import train_and_evaluate


st.set_page_config(page_title="Sentiment Evaluation", layout="wide")
st.title("Sentiment Analysis - TextBlob Labeling and Naive Bayes Evaluation")


def plot_confusion_matrix_annotated(cf_matrix, title="Confusion Matrix (Annotated)"):
    cf = np.array(cf_matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(cf)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    total = cf.sum()
    if total == 0:
        total = 1

    if cf.shape == (2, 2):
        group_names = np.array([["True Neg", "False Pos"],
                                ["False Neg", "True Pos"]])

        for i in range(2):
            for j in range(2):
                count = cf[i, j]
                pct = count / total
                text = f"{group_names[i, j]}\n{count}\n{pct:.2%}"
                ax.text(j, i, text, ha="center", va="center")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
    else:
        for (i, j), v in np.ndenumerate(cf):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks(range(cf.shape[1]))
        ax.set_yticks(range(cf.shape[0]))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def add_textblob_sentiment(df: pd.DataFrame, text_col: str):
    work = df.copy()
    work[text_col] = work[text_col].astype(str)

    polarity = lambda x: TextBlob(x).sentiment.polarity
    work["polarity"] = work[text_col].apply(polarity)

    def analysis(score):
        if score > 0:
            return "positive"
        else:
            return "negative"

    work["Score"] = work["polarity"].apply(analysis)
    return work


def plot_pie_score(df: pd.DataFrame, label_col: str = "Score"):
    vc = df[label_col].value_counts()
    sizes = vc.values
    labels = vc.index.tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    explode = [0.01] * len(labels)
    ax.pie(
        x=sizes,
        labels=labels,
        autopct="%1.1f%%",
        explode=explode,
        textprops={"fontsize": 12},
    )
    ax.set_title("Sentiment Polarity on Tweets Data", fontsize=16, pad=20)
    fig.tight_layout()
    return fig


def auto_pick_text_col(columns):
    candidates = ["Clean Tweet"]
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


uploaded = st.file_uploader("Upload dataset (CSV) dan pastikan kolom bernama Clean Tweet serta data dalam bahasa Inggris", type=["csv"])
if not uploaded:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip()

st.subheader("Preview Data")
st.write("Kolom yang terbaca:", list(df.columns))
st.dataframe(df.head(10), use_container_width=True)

text_col = auto_pick_text_col(list(df.columns))
if text_col is None:
    st.error("Kolom teks tidak ditemukan otomatis. Pastikan ada kolom seperti: Clean Tweet.")
    st.write("Kolom yang tersedia:", list(df.columns))
    st.stop()

st.divider()

st.subheader("TextBlob Polarity and Labeling")

if st.button("Hitung polarity dan buat Score (positive/negative)", type="primary"):
    try:
        with st.spinner("Memproses TextBlob..."):
            df_tb = add_textblob_sentiment(df, text_col=text_col)

        st.write("Data setelah ditambah kolom polarity dan Score")
        st.dataframe(df_tb.head(10), use_container_width=True)

        pos_count = int((df_tb["Score"] == "positive").sum())
        neg_count = int((df_tb["Score"] == "negative").sum())

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Positif", pos_count)
        with c2:
            st.metric("Negatif", neg_count)
        with c3:
            st.write("Distribusi Score")
            fig_pie = plot_pie_score(df_tb, label_col="Score")
            st.pyplot(fig_pie, use_container_width=True)

        st.session_state["df_tb"] = df_tb

    except Exception as e:
        st.error("Gagal memproses TextBlob. Detail:")
        st.exception(e)
        st.stop()

st.divider()

st.subheader("Naive Bayes Evaluation (TF-IDF + SMOTE)")

if "df_tb" not in st.session_state:
    st.warning("Silakan jalankan proses TextBlob terlebih dahulu agar label Score tersedia.")
    st.stop()

model_df = st.session_state["df_tb"]
label_col = "Score"

test_size = st.selectbox("Pilih test_size", options=[0.1, 0.2, 0.3], index=0)

if st.button("Train and Evaluate Naive Bayes"):
    try:
        with st.spinner("Training dan evaluasi model..."):
            results = train_and_evaluate(
                df=model_df,
                text_col=text_col,
                label_col=label_col,
                test_size=float(test_size),
                random_state=42
            )

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader(f"Confusion Matrix (Table) - test_size={test_size}")
            st.dataframe(results["cf_matrix"], use_container_width=True)

            st.subheader("Confusion Matrix Heatmap (Annotated)")
            fig_cm = plot_confusion_matrix_annotated(
                results["cf_matrix"],
                title=f"Confusion Matrix (test_size={test_size})"
            )
            st.pyplot(fig_cm, use_container_width=True)

        with c2:
            st.subheader("Classification Report (Text)")
            st.code(results["report_text"], language="text")

    except Exception as e:
        st.error("Terjadi error saat training/evaluasi. Detail:")
        st.exception(e)
        st.stop()