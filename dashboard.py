"""
Dashboard EDA — Deteksi Phishing SMS Indonesia
Pastikan file 'dataset_phishing_sms_indo.csv' berada di folder yang sama.
"""

# Import Library
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from collections import Counter
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
 
warnings.filterwarnings("ignore")
 
# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA · Phishing SMS Indonesia",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Palet Warna ───────────────────────────────────────────────────────────────
COLOR_PHISHING = "#F44336"
COLOR_NORMAL   = "#2196F3"
COLOR_ACCENT   = "#FF9800"
PALETTE_LABEL  = {0: COLOR_NORMAL, 1: COLOR_PHISHING}
 
plt.rcParams.update({
    "figure.dpi"        : 110,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.size"         : 10,
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
})
 
# ── Load & Preprocessing Data ────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(int)
    df["label_nama"] = df["label"].map({0: "Normal", 1: "Phishing"})

    # Fitur numerik — hitung ulang dari teks agar selalu konsisten
    df["text_length"]       = df["teks"].str.len()
    df["word_count"]        = df["teks"].str.split().str.len()
    df["exclamation_count"] = df["teks"].str.count("!")
    df["question_count"]    = df["teks"].str.count(r"\?")
    df["uppercase_ratio"]   = df["teks"].apply(
        lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1)
    )
    df["digit_ratio"] = df["teks"].apply(
        lambda x: sum(c.isdigit() for c in str(x)) / max(len(str(x)), 1)
    )
    df["avg_word_length"] = df["teks"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
    )

    # Fitur biner — hitung ulang dari teks
    POLA_URL    = re.compile(r"http[s]?://\S+|www\.\S+|bit\.ly/\S+|tsel\.me/\S+", re.I)
    POLA_NOMOR  = re.compile(r"\b\d{8,}\b")
    POLA_KODE   = re.compile(r"\b\d{4,7}\b")
    POLA_RUPIAH = re.compile(r"rp\.?\s?\d|rupiah|juta|ribu", re.I)

    df["has_url"]      = df["teks"].str.contains(POLA_URL,    na=False).astype(int)
    df["has_phone"]    = df["teks"].str.contains(POLA_NOMOR,  na=False).astype(int)
    df["has_code"]     = df["teks"].str.contains(POLA_KODE,   na=False).astype(int)
    df["has_currency"] = df["teks"].str.contains(POLA_RUPIAH, na=False).astype(int)
    return df
 
 
# ── Helper ────────────────────────────────────────────────────────────────────
def fig_to_streamlit(fig, caption: str = ""):
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)
 
 
def metric_row(cols_data: list[tuple]):
    """Render satu baris st.metric dari list (label, value, delta?)."""
    cols = st.columns(len(cols_data))
    for col, item in zip(cols, cols_data):
        if len(item) == 2:
            col.metric(item[0], item[1])
        else:
            col.metric(item[0], item[1], item[2])
 
 
def insight_box(text: str):
    st.info(f"💡 **Insight:** {text}")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 📩🔐")
    st.title("Dashboard EDA - Deteksi Pesan Berhadiah Phishing")
    st.markdown("---")
    sections = [
        "🏠 Overview & Statistik Deskriptif",
        "📊 BQ1 · Distribusi & Keseimbangan Kelas",
        "🎯 BQ2 · Modus Phishing Apa yang Paling Dominan?",
        "🔗 BQ3 · URL, Nomor & Fitur Struktural",
        "📝 BQ4 · Kata & Frasa Dominan",
        "📏 BQ5 · Panjang Teks & Uji Statistik",
    ]
    section = st.radio("Pilih Bagian:", sections, label_visibility="collapsed")
    st.markdown("---")
 
# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df = load_data("dataset_phishing_sms_indo.csv")
except FileNotFoundError:
    st.error(
        f"⚠️ File tidak ditemukan. "
        "Pastikan CSV berada di folder yang sama dengan script ini, "
        "atau ubah path di sidebar."
    )
    st.stop()
 
df_phishing = df[df["label"] == 1].copy()
df_normal   = df[df["label"] == 0].copy()
 
# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER UTAMA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📩🔐 EDA · Deteksi Phishing SMS Berbahasa Indonesia")
st.markdown(
    "Dashboard ini menyajikan Exploratory Data Analysis (EDA) untuk dataset "
    "**Phishing SMS Bahasa Indonesia**, mencakup statistik deskriptif dan "
    "jawaban atas 5 Business Questions (BQ) inti."
)
st.markdown("---")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 0 — OVERVIEW & STATISTIK DESKRIPTIF
# ═══════════════════════════════════════════════════════════════════════════════
if section == sections[0]:
    st.header("🏠 Overview & Statistik Deskriptif")
 
    # KPI cards
    total = len(df)
    n_phishing = (df["label"] == 1).sum()
    n_normal   = (df["label"] == 0).sum()
    rasio      = n_phishing / n_normal
    imbalance_ratio = max(n_phishing, n_normal) / min(n_phishing, n_normal)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Data",            f"{total:,}")
    col2.metric("Phishing",              f"{n_phishing:,}")
    col2.markdown(f"""
        <p style='background-color:#F44336; color:white; font-size:12px;
        padding:2px 7px; border-radius:6px; display:inline-block;
        position:relative; top:-15px;'>
            {n_phishing/total*100:.1f}%
        </p>
    """, unsafe_allow_html=True)
    col3.metric("Normal",                f"{n_normal:,}")
    col3.markdown(f"""
        <p style='background-color:#2196F3; color:white; font-size:12px;
        padding:2px 7px; border-radius:6px; display:inline-block;
        position:relative; top:-15px;'>
            {n_normal/total*100:.1f}%
        </p>
    """, unsafe_allow_html=True)  
    col4.metric("Rasio Phishing:Normal", f"{rasio:.2f}:1")
    col5.metric("Imbalance Ratio", "1.52 : 1")
    col5.markdown(f"""
        <p style='font-size:11px; color:gray; position:relative; top:-15px;'>
            Dataset ini didominasi oleh pesan phishing
        </p>
    """, unsafe_allow_html=True)
 
    # Info kolom
    st.subheader("Informasi Kolom Dataset")
    col_info = pd.DataFrame({
        "Kolom"      : ["teks", "teks_bersih", "teks_processed", "label", "sub_kategori", "sumber"],
        "Tipe"       : ["str", "str", "str", "int", "str", "str"],
        "Keterangan" : [
            "Teks pesan asli",
            "Teks setelah cleaning dasar",
            "Teks setelah preprocessing NLP lengkap",
            "0 = Normal, 1 = Phishing",
            "Tag modus phishing (pipe-separated, mis. berhadiah|phishing_url|minta_data)",
            "Asal dataset (yudiwbs_gist, kaggle_email_indo, bopbi_*, dll.)",
        ],
        "Non-Null"   : [df[c].notna().sum() for c in ["teks", "teks_bersih", "teks_processed", "label", "sub_kategori", "sumber"]],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)
 
    st.markdown("---")
 
    # Statistik deskriptif fitur numerik
    st.subheader("Statistik Deskriptif Fitur Numerik")
    FITUR_NUM = ["text_length", "word_count", "exclamation_count",
                 "question_count", "uppercase_ratio", "digit_ratio",
                 "avg_word_length"]
 
    tab_all, tab_phishing, tab_normal = st.tabs(["Semua Data", "Pesan Phishing", "Pesan Normal"])
    with tab_all:
        st.dataframe(df[FITUR_NUM].describe().round(3), use_container_width=True)
    with tab_phishing:
        st.dataframe(df_phishing[FITUR_NUM].describe().round(3), use_container_width=True)
    with tab_normal:
        st.dataframe(df_normal[FITUR_NUM].describe().round(3), use_container_width=True)
 
    st.markdown("---")
 
    # Distribusi sumber
    st.subheader("Distribusi Sumber Dataset")
    sumber_dist = df.groupby(["sumber", "label_nama"]).size().unstack(fill_value=0)
    for col in ["Normal", "Phishing"]:
        if col not in sumber_dist.columns:
            sumber_dist[col] = 0
 
    fig, ax = plt.subplots(figsize=(10, 4))
    sumber_dist[["Normal", "Phishing"]].plot(
        kind="bar", ax=ax, color=[COLOR_NORMAL, COLOR_PHISHING],
        edgecolor="white", linewidth=0.8, width=0.7
    )
    ax.set_title("Distribusi Label per Sumber Dataset", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah Data")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Kelas", loc="upper right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=8, padding=2)
    plt.tight_layout()
    fig_to_streamlit(fig)
 
    st.dataframe(
        df.groupby("sumber")["label"].value_counts(normalize=True)
          .mul(100).round(1).rename("Persentase (%)").reset_index()
          .rename(columns={"label": "Kelas"})
          .replace({0: "Normal", 1: "Phishing"}),
        use_container_width=True, hide_index=True
    )
 
    st.subheader("Preview Data per Kelas")
    tab_prev_normal, tab_prev_phishing = st.tabs(["Normal", "Phishing"])

    with tab_prev_normal:
        st.dataframe(
            df_normal[["teks", "sub_kategori", "sumber"]].head(10).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

    with tab_prev_phishing:
        sub_kat_list = df_phishing["sub_kategori"].unique().tolist()
        sub_kat_list = ["Semua"] + sorted(sub_kat_list)
        filter_kat = st.selectbox("Filter sub_kategori:", sub_kat_list, key="preview_subkat")

        if filter_kat == "Semua":
            df_preview = df_phishing
        else:
            df_preview = df_phishing[df_phishing["sub_kategori"] == filter_kat]

        st.dataframe(
            df_preview[["teks", "sub_kategori", "sumber"]].head(10).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )
         
# ═══════════════════════════════════════════════════════════════════════════════
#  BQ1 — DISTRIBUSI & KESEIMBANGAN KELAS
# ═══════════════════════════════════════════════════════════════════════════════
elif section == sections[1]:
    st.header("📊 BQ1 · Seberapa Seimbang/Proporsional Data yang Dimiliki?")
 
    st.markdown(
        """
        **Pertanyaan Bisnis:** Seberapa akurat model nantinya dapat membedakan pesan phishing dan normal?  
        Salah satu prasyarat awal adalah mengetahui apakah distribusi kelas **seimbang** atau **imbalanced**,
        karena ketidakseimbangan kelas yang ekstrem dapat menyebabkan model bias ke kelas mayoritas.
        """
    )
 
    st.markdown("---")
 
    # ── Metrik ───────────────────────────────────────────────────────────────
    st.subheader("📌 Metrik Utama")
    total      = len(df)
    n_phishing = (df["label"] == 1).sum()
    n_normal   = (df["label"] == 0).sum()
    rasio      = n_phishing / n_normal
    imbalance_ratio = max(n_phishing, n_normal) / min(n_phishing, n_normal)
 
    metric_row([
        ("Total Data",            f"{total:,}"),
        ("Phishing",              f"{n_phishing:,}",  f"{n_phishing/total*100:.1f}%"),
        ("Normal",                f"{n_normal:,}",    f"{n_normal/total*100:.1f}%"),
        ("Rasio Phishing:Normal", f"{rasio:.2f}:1"),
        ("Imbalance Ratio",       f"{imbalance_ratio:.2f}x"),
    ])
 
    st.markdown("---")
 
    # ── Visualisasi Bar + Donut ───────────────────────────────────────────────
    st.subheader("📊 Visualisasi Distribusi Kelas")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribusi Kelas Dataset Phishing SMS Indonesia",
                 fontsize=13, fontweight="bold")
 
    # Bar chart
    ax = axes[0]
    counts = [n_normal, n_phishing]
    bars = ax.bar(["Normal", "Phishing"], counts,
                  color=[COLOR_NORMAL, COLOR_PHISHING],
                  width=0.5, edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Jumlah Data per Kelas")
    ax.set_ylabel("Jumlah Data")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
    # Donut chart
    ax2 = axes[1]
    wedge_props = dict(width=0.55, edgecolor="white", linewidth=2)
    wedges, texts, autotexts = ax2.pie(
        counts, labels=["Normal", "Phishing"],
        colors=[COLOR_NORMAL, COLOR_PHISHING],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=wedge_props, pctdistance=0.75,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
    ax2.set_title("Proporsi Kelas (Donut Chart)")
    ax2.text(0, 0, f"Total\n{total:,}", ha="center", va="center",
             fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig_to_streamlit(fig)
 
    # ── Tabel ringkasan ───────────────────────────────────────────────────────
    st.subheader("📋 Tabel Distribusi Kelas")
    tabel = pd.DataFrame({
        "Kelas"         : ["Normal", "Phishing", "Total"],
        "Jumlah"        : [n_normal, n_phishing, total],
        "Persentase (%)" : [
            f"{n_normal/total*100:.1f}",
            f"{n_phishing/total*100:.1f}",
            "100.0",
        ],
    })
    st.dataframe(tabel, use_container_width=True, hide_index=True)
 
    st.markdown("---")
    st.subheader("🔍 Penjelasan & Insight")
    insight_box(
        f"Dataset ini berisikan sebanyak 1.772 pesan, dengan didominasi oleh **pesan phishing** sebanyak **{n_phishing:,} ({n_phishing/total*100:.1f}%)** "
        f"serta **{n_normal:,}** lainnya merupakan **pesan normal ({n_normal/total*100:.1f}%)**. "
        f"Sehingga, didapatkan nilai Imbalance ratio sebesar **{imbalance_ratio:.2f}x** — tergolong *ringan* dan masih dalam "
        f"batas aman (threshold umum: >2x mulai bermasalah). "
        f"Distribusi dataset ini masih tergolong cukup baik untuk pelatihan model klasifikasi deteksi pesan phishing "
        f"Maka dari itu, tetap disarankan untuk memantau F1-Score dan AUC-ROC agar model tidak bias ke kelas mayoritas (pesan phishing)"
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  BQ2 — MODUS & TAKTIK PHISHING
# ═══════════════════════════════════════════════════════════════════════════════
elif section == sections[2]:
    st.header("🎯 BQ2 · Modus Phishing Apa yang Paling Dominan?")
 
    st.markdown(
        """
        **Pertanyaan Bisnis:** Modus phishing apa yang paling dominan, khususnya kategori *berhadiah*?  
        Memahami taktik yang digunakan pelaku membantu prioritisasi perlindungan dan penyusunan 
        aturan deteksi berbasis konten.
        """
    )
 
    st.markdown("---")
 
    # ── Explode tag ───────────────────────────────────────────────────────────
    df_ph = df_phishing.copy()
    df_ph["kategori_list"] = df_ph["sub_kategori"].str.split("|")
    exploded = df_ph.explode("kategori_list")
    tag_counts = exploded["kategori_list"].value_counts()
 
    n_berhadiah = df_ph["sub_kategori"].str.contains("berhadiah", na=False).sum()
    n_minta_data = df_ph["sub_kategori"].str.contains("minta_data", na=False).sum()
    n_url        = df_ph["sub_kategori"].str.contains("phishing_url", na=False).sum()
    n_urgensi    = df_ph["sub_kategori"].str.contains("urgensi", na=False).sum()
 
    # ── Metrik ───────────────────────────────────────────────────────────────
    st.subheader("📌 Metrik Utama")
    metric_row([
        ("Total Phishing",    f"{len(df_ph):,}"),
        ("Tag: minta_data",   f"{tag_counts.get('minta_data', 0):,}",
                              f"{tag_counts.get('minta_data',0)/len(df_ph)*100:.1f}%"),
        ("Tag: phishing_url", f"{tag_counts.get('phishing_url', 0):,}",
                              f"{tag_counts.get('phishing_url',0)/len(df_ph)*100:.1f}%"),
        ("Tag: urgensi",      f"{tag_counts.get('urgensi', 0):,}",
                              f"{tag_counts.get('urgensi',0)/len(df_ph)*100:.1f}%"),
        ("Tag: berhadiah",    f"{tag_counts.get('berhadiah', 0):,}",
                              f"{tag_counts.get('berhadiah',0)/len(df_ph)*100:.1f}%"),
    ])
 
    st.markdown("---")
    st.subheader("📊 Visualisasi Frekuensi Tag Taktik Phishing")
 
    col1, col2 = st.columns(2)
 
    # Bar chart top tags
    with col1:
        top_tags = tag_counts.head(8)
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_bar = [COLOR_PHISHING for t in top_tags.index]
        bars = ax.barh(top_tags.index[::-1], top_tags.values[::-1],
                       color=colors_bar[::-1], edgecolor="white", linewidth=0.8)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 10, bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", fontsize=9)
        ax.set_title("Distribusi Tag Taktik Phishing\n(Kemunculan per Pesan)", fontweight="bold")
        ax.set_xlabel("Frekuensi")
        plt.tight_layout()
        fig_to_streamlit(fig)
 
    # Treemap kombinasi sub_kategori terbanyak
    with col2:
        top_kombo = df_ph["sub_kategori"].value_counts().reset_index()
        top_kombo.columns = ["Kombinasi Taktik", "Jumlah"]
        top_kombo = top_kombo[top_kombo["Kombinasi Taktik"] != "lainnya"].head(8).reset_index(drop=True)
        
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.barplot(
            data=top_kombo, y="Kombinasi Taktik", x="Jumlah",
            palette="Reds_r", ax=ax2,
            width=0.75,
        )
        ax2.set_title("Top 8 Kombinasi Taktik Phishing", fontweight="bold")
        ax2.set_xlabel("Jumlah Pesan")
        ax2.set_ylabel("")
        for bar in ax2.patches:
            w = bar.get_width()
            ax2.text(w + 1, bar.get_y() + bar.get_height() / 2,
                     f"{int(w)}", va="center", fontsize=8)
        plt.tight_layout()
        fig_to_streamlit(fig2)
 
    # ── Tabel ────────────────────────────────────────────────────────────────
    st.subheader("📋 Tabel Frekuensi Tag Taktik")
    tabel_tag = tag_counts.reset_index()
    tabel_tag.columns = ["Taktik", "Frekuensi"]
    tabel_tag["% dari Total Phishing"] = (tabel_tag["Frekuensi"] / len(df_ph) * 100).round(1)
    st.dataframe(tabel_tag, use_container_width=True, hide_index=True)
 
    st.markdown("---")
    st.subheader("🔍 Penjelasan & Insight")
    insight_box(
        f"Pesan Berjenis **minta_data** merupakan pesan yang paling dominan pada dataset ini, dengan total sebanyak "
        f"{tag_counts.get('minta_data',0):,} pesan "
        f"({tag_counts.get('minta_data',0)/len(df_ph)*100:.1f}%), "
        f"diikuti oleh **phishing_url** sebanyak {tag_counts.get('phishing_url',0):,} pesan "
        f"({tag_counts.get('phishing_url',0)/len(df_ph)*100:.1f}%), "        
        f"dan **berhadiah** sebanyak ({tag_counts.get('berhadiah',0):,}) pesan "
        f"({tag_counts.get('berhadiah',0)/len(df_ph)*100:.1f}%), "
        f"Adapun kombinasi taktik pesan phishing yang paling umum adalah kombinasi pesan berhadiah, tautan phishing, serta meminta data pribadi, diikuti oleh pesan yang hanya meminta data pribadi serta pesan yang hanya mencantumkan tautan phishing. "
        f"Hal ini menunjukkan bahwa taktik meminta data pribadi, memberikan tautan phising, serta beberapa kombinasi taktik phising cenderung sering dilakukan untuk menipu banyak orang. "
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  BQ3 — URL, NOMOR & FITUR STRUKTURAL
# ═══════════════════════════════════════════════════════════════════════════════
elif section == sections[3]:
    st.header("🔗 BQ3 · Apakah URL, Nomor Panjang & Fitur Lain Menjadi Indikator Kuat Phishing?")
 
    st.markdown(
        """
        **Pertanyaan Bisnis:** Apakah keberadaan URL, nomor telepon panjang, kode OTP, 
        atau fitur struktural lain berkorelasi signifikan dengan label phishing?  
        Ini menentukan fitur mana yang paling informatif untuk model klasifikasi.
        """
    )
 
    st.markdown("---")
 
    FITUR_ALL = [
        "text_length", "word_count", "exclamation_count", "question_count",
        "uppercase_ratio", "digit_ratio", "avg_word_length",
        "has_url", "has_phone", "has_code", "has_currency",
    ]
    corr_series = df[FITUR_ALL + ["label"]].corr()["label"].drop("label").sort_values(
        key=abs, ascending=False
    )
 
    # ── Metrik ───────────────────────────────────────────────────────────────
    st.subheader("📌 Metrik Utama")
    pct_url_phishing   = df_phishing["has_url"].mean() * 100
    pct_url_normal     = df_normal["has_url"].mean() * 100
    pct_phone_phishing = df_phishing["has_phone"].mean() * 100
    pct_phone_normal   = df_normal["has_phone"].mean() * 100
    pct_code_phishing  = df_phishing["has_code"].mean() * 100
    pct_code_normal    = df_normal["has_code"].mean() * 100
 
    metric_row([
        ("URL di Phishing",    f"{pct_url_phishing:.1f}%"),
        ("URL di Normal",      f"{pct_url_normal:.1f}%"),
        ("Nomor di Phishing",  f"{pct_phone_phishing:.1f}%"),
        ("Nomor di Normal",    f"{pct_phone_normal:.1f}%"),
        ("Kode di Phishing",   f"{pct_code_phishing:.1f}%"),
        ("Kode di Normal",     f"{pct_code_normal:.1f}%"),
    ])
 
    st.markdown("---")
 
    col1, col2 = st.columns(2)
 
    # Heatmap korelasi
    with col1:
        st.subheader("Korelasi Fitur vs Label")
        fig, ax = plt.subplots(figsize=(5, 5))
        colors_bar2 = [COLOR_PHISHING if v > 0 else COLOR_NORMAL
                       for v in corr_series.values]
        bars = ax.barh(corr_series.index[::-1], corr_series.values[::-1],
                       color=colors_bar2[::-1], edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Korelasi Pearson Fitur → Label\n(+ = cenderung Phishing)", fontweight="bold")
        ax.set_xlabel("Korelasi Pearson")
        THRESHOLD = 0.08  # bar lebih panjang dari ini → angka di dalam
        for bar in bars:
            w = bar.get_width()
            if abs(w) >= THRESHOLD:
                # Angka di dalam bar
                x_pos = w / 2
                ha = "center"
                color_txt = "white"
            else:
                # Angka di luar bar
                x_pos = w + 0.005 if w >= 0 else w - 0.005
                ha = "left" if w >= 0 else "right"
                color_txt = "black"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", ha=ha, fontsize=8, color=color_txt)
        plt.tight_layout()
        fig_to_streamlit(fig)
 
    # Bar grouped fitur biner
    with col2:
        st.subheader("Prevalensi Fitur Biner per Kelas (%)")
        biner_feats = ["has_url", "has_phone", "has_code", "has_currency"]
        pct_phishing = [df_phishing[f].mean() * 100 for f in biner_feats]
        pct_normal_  = [df_normal[f].mean() * 100 for f in biner_feats]
        labels_b     = ["URL", "Nomor Panjang", "Kode Numerik", "Mata Uang"]
 
        x = np.arange(len(labels_b))
        width = 0.35
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bars1 = ax2.bar(x - width / 2, pct_normal_,   width, label="Normal",   color=COLOR_NORMAL,   edgecolor="white")
        bars2 = ax2.bar(x + width / 2, pct_phishing, width, label="Phishing", color=COLOR_PHISHING, edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_b, fontsize=9)
        ax2.set_ylabel("Persentase (%)")
        ax2.set_title("Prevalensi Fitur Biner\nper Kelas (%)", fontweight="bold")
        ax2.legend()
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        fig_to_streamlit(fig2)
 
    # Statistik deskriptif fitur per kelas
    st.subheader("📋 Statistik Deskriptif Fitur Numerik per Kelas")
    tabel_stat = (
        df.groupby("label_nama")[
            ["text_length", "word_count", "exclamation_count",
             "uppercase_ratio", "digit_ratio", "avg_word_length"]
        ]
        .agg(["mean", "median", "std"])
        .round(3)
    )
    st.dataframe(tabel_stat, use_container_width=True)
 
    # Correlation table
    st.subheader("📋 Tabel Korelasi Fitur → Label")
    corr_df = corr_series.reset_index()
    corr_df.columns = ["Fitur", "Korelasi Pearson"]
    corr_df["Interpretasi"] = corr_df["Korelasi Pearson"].apply(
        lambda v: "➕ Cenderung Phishing" if v > 0 else "➖ Cenderung Normal"
    )
    corr_df["Kekuatan"] = corr_df["Korelasi Pearson"].abs().apply(
        lambda v: "Kuat" if v > 0.3 else ("Sedang" if v > 0.1 else "Lemah")
    )
    st.dataframe(corr_df, use_container_width=True, hide_index=True)
 
    st.markdown("---")
    st.subheader("🔍 Penjelasan & Insight")
    insight_box(
            "Fitur **has_code** memiliki korelasi negatif terkuat (-0.230), menandakan bahwa pesan normal "
            "justru lebih sering mengandung kode numerik, kemungkinan besar berupa kode OTP resmi. "
            "Di sisi phishing, **has_phone** (+0.209) dan **exclamation_count** (+0.194) menjadi "
            "sinyal paling kuat, mencerminkan pola penipuan yang menyertakan nomor kontak palsu "
            "dan nada urgensi berlebihan. "
            "**has_url** (+0.089) juga berkontribusi positif meski tidak dominan — "
            "karena pesan normal pun kadang menyertakan URL resmi. "
            "Fitur seperti **question_count**, **digit_ratio**, dan **uppercase_ratio** hampir tidak "
            "berkorelasi, sehingga kurang informatif jika digunakan sendiri. "
        )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  BQ4 — KATA & FRASA DOMINAN
# ═══════════════════════════════════════════════════════════════════════════════
elif section == sections[4]:
    st.header("📝 BQ4 · Kata dan Frasa Apa yang Paling Sering Muncul di Phishing vs Normal?")
 
    st.markdown(
        """
        **Pertanyaan Bisnis:** Kata dan frasa apa yang paling sering muncul di pesan phishing 
        dibandingkan pesan normal?  
        Analisis ini membantu memahami pola linguistik yang dapat dijadikan fitur untuk deteksi otomatis.
        """
    )
 
    st.markdown("---")
 
    @st.cache_data
    def compute_word_stats(df_in):
        def top_words(series, n=25):
            words = " ".join(series.dropna().astype(str)).split()
            return Counter(words).most_common(n)
 
        ph_words = top_words(df_in[df_in.label == 1]["teks_processed"])
        nm_words = top_words(df_in[df_in.label == 0]["teks_processed"])
 
        # TF-IDF discriminative terms
        vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        vec.fit(df_in["teks_processed"].dropna().astype(str))
        tfidf_ph = TfidfVectorizer(max_features=20, ngram_range=(1, 1))
        tfidf_nm = TfidfVectorizer(max_features=20, ngram_range=(1, 1))
        tfidf_ph.fit(df_in[df_in.label == 1]["teks_processed"].dropna().astype(str))
        tfidf_nm.fit(df_in[df_in.label == 0]["teks_processed"].dropna().astype(str))
 
        ph_tfidf = sorted(
            zip(tfidf_ph.get_feature_names_out(), tfidf_ph.idf_),
            key=lambda x: x[1], reverse=True
        )[:15]
        nm_tfidf = sorted(
            zip(tfidf_nm.get_feature_names_out(), tfidf_nm.idf_),
            key=lambda x: x[1], reverse=True
        )[:15]
 
        return ph_words, nm_words, ph_tfidf, nm_tfidf
 
    ph_words, nm_words, ph_tfidf, nm_tfidf = compute_word_stats(df)
 
    # ── Metrik ───────────────────────────────────────────────────────────────
    st.subheader("📌 Kata Paling Diskriminatif")
    top_phishing_word = ph_words[0][0] if ph_words else "-"
    top_normal_word   = nm_words[0][0] if nm_words else "-"
    metric_row([
        ("Kata #1 Phishing", top_phishing_word, f"Freq: {ph_words[0][1]:,}"),
        ("Kata #1 Normal",   top_normal_word,   f"Freq: {nm_words[0][1]:,}"),
        ("Kata unik Phishing (proses)", f"{len(set(' '.join(df_phishing['teks_processed'].dropna()).split())):,}"),
        ("Kata unik Normal (proses)",   f"{len(set(' '.join(df_normal['teks_processed'].dropna()).split())):,}"),
    ])
 
    st.markdown("---")
 
    tab1, tab2 = st.tabs(["📊 Frekuensi & TF-IDF", "☁️ Word Cloud"])
 
    with tab1:
        col1, col2 = st.columns(2)
        # Frequency bar
        with col1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # Phishing
            ax = axes[0]
            words_ph, freqs_ph = zip(*ph_words[:20])
            ax.barh(list(words_ph)[::-1], list(freqs_ph)[::-1],
                    color=COLOR_PHISHING, edgecolor="white")
            ax.set_title("Top 20 Kata — Phishing", fontweight="bold", color=COLOR_PHISHING)
            ax.set_xlabel("Frekuensi")
            plt.tight_layout()
            # Normal
            ax2 = axes[1]
            words_nm, freqs_nm = zip(*nm_words[:20])
            ax2.barh(list(words_nm)[::-1], list(freqs_nm)[::-1],
                     color=COLOR_NORMAL, edgecolor="white")
            ax2.set_title("Top 20 Kata — Normal", fontweight="bold", color=COLOR_NORMAL)
            ax2.set_xlabel("Frekuensi")
            plt.tight_layout()
            fig_to_streamlit(fig)
 
        # TF-IDF bar
        with col2:
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            # Phishing TF-IDF
            ax = axes2[0]
            t_ph, s_ph = zip(*ph_tfidf)
            ax.barh(list(t_ph)[::-1], list(s_ph)[::-1],
                    color=COLOR_PHISHING, alpha=0.85, edgecolor="white")
            ax.set_title("TF-IDF Top 15 — Phishing", fontweight="bold", color=COLOR_PHISHING)
            ax.set_xlabel("IDF Score")
            # Normal TF-IDF
            ax2 = axes2[1]
            t_nm, s_nm = zip(*nm_tfidf)
            ax2.barh(list(t_nm)[::-1], list(s_nm)[::-1],
                     color=COLOR_NORMAL, alpha=0.85, edgecolor="white")
            ax2.set_title("TF-IDF Top 15 — Normal", fontweight="bold", color=COLOR_NORMAL)
            ax2.set_xlabel("IDF Score")
            plt.tight_layout()
            fig_to_streamlit(fig2)
 
        # Tabel
        st.subheader("📋 Tabel Frekuensi Kata")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Phishing**")
            st.dataframe(
                pd.DataFrame(ph_words[:20], columns=["Kata", "Frekuensi"]),
                use_container_width=True, hide_index=True
            )
        with c2:
            st.markdown("**Normal**")
            st.dataframe(
                pd.DataFrame(nm_words[:20], columns=["Kata", "Frekuensi"]),
                use_container_width=True, hide_index=True
            )
 
    with tab2:
        st.markdown("Word Cloud dibuat dari kolom `teks_processed` (sudah di-stemming).")
        col1, col2 = st.columns(2)
 
        @st.cache_data
        def make_wordcloud(text, color):
            wc = WordCloud(
                width=700, height=400, background_color="white",
                colormap="Reds" if color == "red" else "Blues",
                max_words=150, collocations=False,
            )
            wc.generate(text)
            return wc
 
        text_ph = " ".join(df_phishing["teks_processed"].dropna().astype(str))
        text_nm = " ".join(df_normal["teks_processed"].dropna().astype(str))
 
        with col1:
            wc = make_wordcloud(text_ph, "red")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Word Cloud — Phishing", fontweight="bold", color=COLOR_PHISHING)
            fig_to_streamlit(fig, "Pesan Phishing")
 
        with col2:
            wc2 = make_wordcloud(text_nm, "blue")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.imshow(wc2, interpolation="bilinear")
            ax2.axis("off")
            ax2.set_title("Word Cloud — Normal", fontweight="bold", color=COLOR_NORMAL)
            fig_to_streamlit(fig2, "Pesan Normal")
 
    st.markdown("---")
    st.subheader("🔍 Penjelasan & Insight")
    insight_box(
        "Pada top kata kategori phishing, tautan url menjadi salah satu unsur yang paling sering muncul dalam pesan, diikuti oleh kata 'kirim', 'klik','com', dan 'situs'." \
        "Sedangkan pada kategori pesan normal, kata 'vince','terima','enron','kasih', dan 'ect' menjadi kata yang paling sering muncul dalam teks pesan. "\
        "Hal ini menunjukkan bahwa pesan phishing lebih cenderung mengarahkan penerima pesan untuk melakukan sesuatu yang dapat memberikan informasi rahasia yang penting kepada pengirim pesan."\
        "Berbeda dengan pesan normal, kata-kata yang muncul cenderung lebih baku beserta kata 'terima' dan 'kasih'."\
        "Hal tersebut mendukung hasil TF-IDF yang memperlihatkan bahwa penggunaan kata-kata yang dapat menggaet orang awam untuk membagi informasi pribadi cenderung mengarahkan kepada pesan phishing, serta penggunaan kata-kata yang lebih bersifat korporat mengarahkan kepada pesan normal."
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  BQ5 — PANJANG TEKS & UJI STATISTIK
# ═══════════════════════════════════════════════════════════════════════════════
elif section == sections[5]:
    st.header("📏 BQ5 · Apakah Panjang Teks Pesan Phishing Berbeda Signifikan dengan Normal?")
 
    st.markdown(
        """
        **Pertanyaan Bisnis:** Apakah terdapat perbedaan signifikan secara statistik antara 
        panjang teks (jumlah karakter dan kata) pesan phishing vs normal?  
        Jika ada, panjang teks bisa menjadi fitur tambahan yang berguna dalam model klasifikasi.
        """
    )
 
    st.markdown("---")
 
    phishing_len = df_phishing["text_length"]
    normal_len   = df_normal["text_length"]
    phishing_wc  = df_phishing["word_count"]
    normal_wc    = df_normal["word_count"]
 
    # Mann-Whitney
    u_char, p_char = stats.mannwhitneyu(phishing_len, normal_len, alternative="two-sided")
    u_word, p_word = stats.mannwhitneyu(phishing_wc,  normal_wc,  alternative="two-sided")
 
    # Effect size (rank-biserial)
    n1, n2 = len(phishing_len), len(normal_len)
    rb_char = 1 - (2 * u_char) / (n1 * n2)
 
# ── Metrik ───────────────────────────────────────────────────────────────
    st.subheader("📌 Metrik Utama")
    p_mantissa, p_exponent = f"{p_char:.2e}".split("e")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Median Karakter — Phishing", f"{phishing_len.median():.0f}")
    col_m2.metric("Median Karakter — Normal",   f"{normal_len.median():.0f}")
    col_m3.metric("Mann-Whitney p-value", f"{p_mantissa} × 10⁻²")
    col_m4.metric("Rank-Biserial Effect Size", f"{rb_char:.3f}")
    col_m5.metric("Signifikan?", "✅ Ya" if p_char < 0.05 else "❌ Tidak")
 
    st.markdown("---")
 
    # ── Visualisasi ───────────────────────────────────────────────────────────
    st.subheader("📊 Visualisasi Distribusi Panjang Teks")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distribusi Panjang Teks: Phishing vs Normal", fontweight="bold")
 
    # Histogram karakter
    ax = axes[0]
    ax.hist(normal_len.clip(upper=3000),   bins=50, alpha=0.7, color=COLOR_NORMAL,
            label="Normal",   edgecolor="white", linewidth=0.4)
    ax.hist(phishing_len.clip(upper=3000), bins=50, alpha=0.7, color=COLOR_PHISHING,
            label="Phishing", edgecolor="white", linewidth=0.4)
    ax.axvline(normal_len.median(),   color=COLOR_NORMAL,   linestyle="--", linewidth=1.5,
               label=f"Median Normal {normal_len.median():.0f}")
    ax.axvline(phishing_len.median(), color=COLOR_PHISHING, linestyle="--", linewidth=1.5,
               label=f"Median Phishing {phishing_len.median():.0f}")
    ax.set_title("Histogram Jumlah Karakter\n(clipped ≤3000)")
    ax.set_xlabel("Jumlah Karakter")
    ax.set_ylabel("Frekuensi")
    ax.legend(fontsize=8)
 
    # Boxplot karakter
    ax2 = axes[1]
    data_box = [normal_len.clip(upper=3000).values, phishing_len.clip(upper=3000).values]
    bp = ax2.boxplot(data_box, labels=["Normal", "Phishing"],
                     patch_artist=True, medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(COLOR_NORMAL)
    bp["boxes"][1].set_facecolor(COLOR_PHISHING)
    ax2.set_title("Boxplot Jumlah Karakter\n(clipped ≤3000)")
    ax2.set_ylabel("Jumlah Karakter")
 
    # Histogram jumlah kata
    ax3 = axes[2]
    ax3.hist(normal_wc.clip(upper=500),   bins=50, alpha=0.7, color=COLOR_NORMAL,
             label="Normal",   edgecolor="white", linewidth=0.4)
    ax3.hist(phishing_wc.clip(upper=500), bins=50, alpha=0.7, color=COLOR_PHISHING,
             label="Phishing", edgecolor="white", linewidth=0.4)
    ax3.axvline(normal_wc.median(),   color=COLOR_NORMAL,   linestyle="--", linewidth=1.5)
    ax3.axvline(phishing_wc.median(), color=COLOR_PHISHING, linestyle="--", linewidth=1.5)
    ax3.set_title("Histogram Jumlah Kata\n(clipped ≤500)")
    ax3.set_xlabel("Jumlah Kata")
    ax3.set_ylabel("Frekuensi")
    ax3.legend(fontsize=8)
 
    plt.tight_layout()
    fig_to_streamlit(fig)
 
    # ── Tabel ringkasan statistik ─────────────────────────────────────────────
    st.subheader("📋 Tabel Statistik Panjang Teks per Kelas")
    summary = pd.DataFrame({
        "Metrik"             : ["Count", "Mean", "Median", "Std", "Min", "Max", "Q25", "Q75"],
        "Phishing (karakter)": [
            f"{len(phishing_len):,}", f"{phishing_len.mean():.1f}",
            f"{phishing_len.median():.1f}", f"{phishing_len.std():.1f}",
            f"{phishing_len.min()}", f"{phishing_len.max():,}",
            f"{phishing_len.quantile(0.25):.1f}", f"{phishing_len.quantile(0.75):.1f}",
        ],
        "Normal (karakter)"  : [
            f"{len(normal_len):,}", f"{normal_len.mean():.1f}",
            f"{normal_len.median():.1f}", f"{normal_len.std():.1f}",
            f"{normal_len.min()}", f"{normal_len.max():,}",
            f"{normal_len.quantile(0.25):.1f}", f"{normal_len.quantile(0.75):.1f}",
        ],
        "Phishing (kata)"    : [
            f"{len(phishing_wc):,}", f"{phishing_wc.mean():.1f}",
            f"{phishing_wc.median():.1f}", f"{phishing_wc.std():.1f}",
            f"{phishing_wc.min()}", f"{phishing_wc.max():,}",
            f"{phishing_wc.quantile(0.25):.1f}", f"{phishing_wc.quantile(0.75):.1f}",
        ],
        "Normal (kata)"      : [
            f"{len(normal_wc):,}", f"{normal_wc.mean():.1f}",
            f"{normal_wc.median():.1f}", f"{normal_wc.std():.1f}",
            f"{normal_wc.min()}", f"{normal_wc.max():,}",
            f"{normal_wc.quantile(0.25):.1f}", f"{normal_wc.quantile(0.75):.1f}",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
 
    # ── Uji statistik ─────────────────────────────────────────────────────────
    st.subheader("🧪 Hasil Uji Statistik (Mann-Whitney U)")
    st.markdown(
        "> *Mann-Whitney U* dipilih karena distribusi panjang teks tidak normal "
        "(terlihat dari right-skewed histogram di atas)."
    )
    uji_df = pd.DataFrame({
        "Fitur"          : ["Jumlah Karakter", "Jumlah Kata"],
        "U Statistic"    : [f"{u_char:.0f}", f"{u_word:.0f}"],
        "p-value"        : [f"{p_char:.4e}", f"{p_word:.4e}"],
        "Signifikan?"    : [
            "✅ Ya (p < 0.05)" if p_char < 0.05 else "❌ Tidak",
            "✅ Ya (p < 0.05)" if p_word < 0.05 else "❌ Tidak",
        ],
        "Effect Size (rb)": [f"{rb_char:.3f}", "-"],
    })
    st.dataframe(uji_df, use_container_width=True, hide_index=True)
 
    st.markdown("---")
    st.subheader("🔍 Penjelasan & Insight")
    insight_box(
        "Uji Mann-Whitney U menunjukkan adanya perbedaan menarik antara metrik karakter dan kata. "
        f"Pada fitur **Jumlah Karakter**, perbedaan panjang teks bersifat **signifikan secara statistik** "
        f"($p = 3.4912\\times10^{{-2}} < 0.05$). Nilai median karakter pada pesan phishing ({333:.0f}) "
        f"memang sedikit lebih panjang dibandingkan pesan normal ({294:.0f}). Namun, efek pembeda ini "
        f"tergolong lemah karena memiliki **effect size yang sangat kecil** (rank-biserial = -0.059).\n\n"
        f"Sementara itu, pada fitur **Jumlah Kata**, perbedaan antara pesan phishing (median = {51:.0f}) "
        f"dan normal (median = {47:.0f}) dinyatakan **tidak signifikan secara statistik** "
        f"($p = 3.7142\\times10^{{-1}} > 0.05$).\n\n"
        "**Kesimpulan untuk Model:** Meskipun jumlah karakter berbeda signifikan, nilai *effect size* "
        "yang sangat kecil serta tidak signifikannya jumlah kata menunjukkan bahwa fitur panjang teks "
        "**tidak cukup andal jika digunakan sendirian** untuk klasifikasi, sehingga wajib dikombinasikan "
        "dengan fitur berbasis konten (seperti TF-IDF)."
    )
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Dashboard EDA · Deteksi Phishing SMS Berbahasa Indonesia · "
    "Dibuat oleh Capstone Team · CC26-PSU107"
)