import os
import time

import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

st.set_page_config(
    page_title="BiComment",
    page_icon="💬",
    layout="centered"
)

# LOAD CSS
def load_css():
    if os.path.exists("style.css"):
        with open("style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# SESSION STATE
if "username" not in st.session_state:
    st.session_state.username = ""

if "login" not in st.session_state:
    st.session_state.login = False

if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_video" not in st.session_state:
    st.session_state.selected_video = None

# DATA VIDEO
videos = [
    {"id": 1, "title": "Video 1", "url": "https://youtu.be/FIXQQ7X7tZE?si=L6aZDc53BUD-4HXk"},
    {"id": 2, "title": "Video 2", "url": "https://youtu.be/q5x1SNjRQwY?si=kJfoJ-VlLeMGh-L8"},
    {"id": 3, "title": "Video 3", "url": "https://youtu.be/HfVq3fXaQGo?si=HM0tNR82NZmIvw1V"},
    {"id": 4, "title": "Video 4", "url": "https://youtu.be/gLptUhuPxSo?si=rIdcTFecpuUBFHmi"},
    {"id": 5, "title": "Video 5", "url": "https://youtu.be/Ip6pCjWp4lk?si=ZnlpefRX40PesTf2"},
    {"id": 6, "title": "Video 6", "url": "https://youtu.be/TbI06M264uE?si=gKVGXtmJzdKY5YxM"},
    {"id": 7, "title": "Video 7", "url": "https://youtu.be/4zTdGISh43w?si=pQNu3JO0bgAhtQ_m"},
    {"id": 8, "title": "Video 8", "url": "https://youtu.be/Q7u7G1NP-Ao?si=4heqxbxkuBoBXvV_"},
    {"id": 9, "title": "Video 9", "url": "https://youtu.be/fkPOq5P8gZ8?si=XgsaZYP5SSzRhKfy"},
    {"id": 10, "title": "Video 10", "url": "https://youtu.be/dHKJL4yJTDw?si=27Pm-dSbV3egw3Xt"},
]

# PATH FILE KOMENTAR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "komentar.csv")

if not os.path.exists(file_path):
    df_awal = pd.DataFrame(columns=["nama", "komentar", "id_konten"])
    df_awal.to_csv(file_path, index=False)

# LOGIN SEDERHANA
def login():
    st.title("Login BiComment")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username.strip() == "" or password.strip() == "":
            st.warning("Username dan password tidak boleh kosong!")
        else:
            st.session_state.login = True
            st.session_state.username = username.strip()
            st.success("Login berhasil!")
            st.rerun()

# LOAD MODEL 2 TAHAP
@st.cache_resource
def load_model():
    stage1_path = os.path.join(BASE_DIR, "model_bicomment_stage1")
    stage2_path = os.path.join(BASE_DIR, "model_bicomment_stage2")

    tokenizer_stage1 = BertTokenizerFast.from_pretrained(stage1_path)
    model_stage1 = BertForSequenceClassification.from_pretrained(stage1_path)

    tokenizer_stage2 = BertTokenizerFast.from_pretrained(stage2_path)
    model_stage2 = BertForSequenceClassification.from_pretrained(stage2_path)

    model_stage1.eval()
    model_stage2.eval()

    return tokenizer_stage1, model_stage1, tokenizer_stage2, model_stage2


tokenizer_stage1, model_stage1, tokenizer_stage2, model_stage2 = load_model()

print("Stage 1 labels:", model_stage1.config.id2label)
print("Stage 2 labels:", model_stage2.config.id2label)

test = "Lo tuh emang dari sananya ga punya otak kali ya"

inputs = tokenizer_stage1(
    test,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model_stage1(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]

print("Raw logits:", outputs.logits)
print("Probabilities:", probs)
print("Pred index:", torch.argmax(probs).item())

# PREDIKSI PER TAHAP
def predict_stage(text, tokenizer, model, labels):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_label = labels[pred_idx]
    pred_score = probs[pred_idx].item()

    all_scores = {
        labels[i]: probs[i].item()
        for i in range(len(labels))
    }

    return pred_label, pred_score, all_scores

# PREDIKSI KOMENTAR 2 TAHAP
def predict_comment(text):
    label_stage1 = ["positive", "neutral", "negative"]
    label_stage2 = ["hate_speech", "insult", "threat"]

    stage1_label, stage1_score, stage1_scores = predict_stage(
        text,
        tokenizer_stage1,
        model_stage1,
        label_stage1
    )

    hasil = {
        "stage1_label": stage1_label,
        "stage1_score": stage1_score,
        "stage1_scores": stage1_scores,
        "stage2_label": None,
        "stage2_score": None,
        "stage2_scores": None,
        "final_label": stage1_label,
        "final_score": stage1_score,
        "status": "ditampilkan"
    }

    if stage1_label in ["negative", "neutral"]:
        stage2_label, stage2_score, stage2_scores = predict_stage(
            text,
            tokenizer_stage2,
            model_stage2,
            label_stage2
            )

        hasil["stage2_label"] = stage2_label
        hasil["stage2_score"] = stage2_score
        hasil["stage2_scores"] = stage2_scores
        hasil["final_label"] = stage2_label
        hasil["final_score"] = stage2_score

        if stage2_label in ["insult", "threat", "hate_speech"]:
            hasil["status"] = "ditolak"
            
        else:
            hasil["status"] = "ditampilkan"
    
    else:
        hasil["status"] = "ditampilkan"

    return hasil

# SIMPAN KOMENTAR
def save_comment(nama, komentar, id_konten):
    new_data = pd.DataFrame(
        [[nama, komentar, str(id_konten)]],
        columns=["nama", "komentar", "id_konten"]
    )

    if os.path.exists(file_path):
        df_lama = pd.read_csv(file_path)
    else:
        df_lama = pd.DataFrame(columns=["nama", "komentar", "id_konten"])

    df_baru = pd.concat([df_lama, new_data], ignore_index=True)
    df_baru.to_csv(file_path, index=False)

# HALAMAN HOME
def show_home():
    st.markdown("<h1 class='main-title'>BiComment</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Pilih konten untuk melihat dan memberi komentar</p>",
        unsafe_allow_html=True
    )

    st.write(f"👤 Login sebagai: {st.session_state.username}")

    for video in videos:
        st.markdown('<div class="video-card">', unsafe_allow_html=True)
        st.subheader(video["title"])
        st.video(video["url"])

        if st.button(f"💬 Lihat & Komentar - {video['id']}", key=f"btn_{video['id']}"):
            st.session_state.page = "detail"
            st.session_state.selected_video = video
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

# HALAMAN DETAIL
def show_detail():
    video = st.session_state.selected_video

    if video is None:
        st.session_state.page = "home"
        st.rerun()

    st.write(f"👤 {st.session_state.username}")
    st.title(video["title"])
    st.video(video["url"])

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.page = "home"
        st.session_state.selected_video = None
        st.rerun()

    with st.form(key="form_komen", clear_on_submit=True):
        komentar = st.text_area("Tulis komentar kamu:")
        submit = st.form_submit_button("Kirim Komentar")

        if submit:
            if not komentar.strip():
                st.warning("Komentar tidak boleh kosong!")
            else:
                with st.spinner("Sedang menganalisis komentar..."):
                    time.sleep(1)
                    hasil = predict_comment(komentar)

                st.subheader("Hasil Deteksi Model")

                if hasil["status"] == "ditolak":
                    st.error(
                        f"❌ Komentar ditolak karena terdeteksi sebagai "
                        f"{hasil['final_label']} ({hasil['final_score'] * 100:.1f}%)."
                    )
                else:
                    save_comment(
                        st.session_state.username,
                        komentar,
                        video["id"]
                    )
                    st.success("✅ Komentar berhasil diposting.")

    st.subheader("💬 Komentar")

    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = pd.DataFrame(columns=["nama", "komentar", "id_konten"])

    komentar_video = df[df["id_konten"].astype(str) == str(video["id"])]

    if komentar_video.empty:
        st.write("Belum ada komentar.")
    else:
        for _, row in komentar_video.iterrows():
            st.write(f"👤 {row['nama']}")
            st.write(f"💬 {row['komentar']}")
            st.markdown("---")

# HALAMAN TENTANG
def show_about():
    st.title("Tentang BiComment")

    st.subheader("Fitur Utama")
    st.write(
        "BiComment adalah aplikasi berbasis web yang dirancang untuk "
        "mendeteksi dan mencegah komentar kasar secara otomatis menggunakan model IndoBERT."
    )
    st.write("- Deteksi komentar kategori insult, hate speech, dan threat.")
    st.write("- Penyimpanan komentar per video.")
    st.write("- Simulasi komentar seperti media sosial.")

    st.subheader("Teknologi")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- IndoBERT")
    st.write("- PyTorch")

    st.subheader("Pengembang")
    st.write("Nama: Sabrina Phalosa Phai")
    st.write("Dosen Pembimbing: Desi Arisandi S.Kom., M.T.I")

# MAIN APP
if not st.session_state.login:
    st.warning("⚠️ Anda harus login terlebih dahulu untuk mengakses BiComment.")
    login()
    st.stop()

st.sidebar.write(f"👤 User: {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

menu = st.sidebar.selectbox("Menu", ["Home", "Tentang"])

if menu == "Home":
    if st.session_state.page == "home":
        show_home()
    else:
        show_detail()
else:
    show_about()