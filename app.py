import os
import glob
import streamlit as st
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="ระบบผู้ช่วยแนะนำการเลือกอาหารเพื่อสุขภาพ",
    page_icon="🥗",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap');

    html, body, .stApp {
        font-family: 'Prompt', sans-serif !important;
        background-color: #F8F9F6 !important;
    }

    .stChatMessage {
        border-radius: 16px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
    }

    [data-testid="stChatMessageAvatarUser"], 
    div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
        background-color: #5B7065 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #2D4A3E !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stChatInput"] {
        border-radius: 20px !important;
        border: 1px solid #D0DDD5 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
    }

    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8E4 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    }

    .cozy-card {
        background: linear-gradient(135deg, #EAF1EC 0%, #F4F7F5 100%);
        border: 1px solid #D2E0D6;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(45, 74, 62, 0.05);
    }
    .cozy-title {
        color: #2D4A3E;
        font-size: 26px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .cozy-subtitle {
        color: #5B7065;
        font-size: 14px;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cozy-card">
    <div class="cozy-title">🥗 ระบบผู้ช่วยแนะนำการเลือกอาหารเพื่อสุขภาพ</div>
    <div class="cozy-subtitle">ระบบตอบคำถามและแนะนำโภชนาการ</div>
</div>
""", unsafe_allow_html=True)

if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ ไม่พบ GEMINI_API_KEY ใน st.secrets กรุณาตั้งค่าใน Streamlit Cloud Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource(show_spinner="กำลังโหลดโมเดลภาษาและสร้าง Vector Database...")
def setup_rag_system():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    chunks = []
    metadata = []
    
    data_folder = "data"
    file_paths = glob.glob(os.path.join(data_folder, "*.txt"))
    
    if not file_paths:
        st.error("❌ ไม่พบไฟล์เอกสารในโฟลเดอร์ data/")
        st.stop()

    chunk_size = 350
    chunk_overlap = 60

    for path in file_paths:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        text = " ".join(text.split())
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(chunk_text)
                metadata.append({"source": filename, "content": chunk_text})
            start += (chunk_size - chunk_overlap)

    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    return model, index, metadata

embed_model, vector_index, doc_metadata = setup_rag_system()

SIMILARITY_THRESHOLD = 0.30

def search_docs(query, k=4):
    query_vector = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = vector_index.search(query_vector.astype('float32'), k)
    
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < len(doc_metadata) and score >= SIMILARITY_THRESHOLD:
            results.append((doc_metadata[idx], score))
            
    return results

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ! ผมคือระบบผู้ช่วยแนะนำการเลือกอาหารเพื่อสุขภาพ มีข้อสงสัยด้านโภชนาการ การอ่านฉลาก หรือการเลือกทานอาหาร สอบถามได้เลยครับ", "avatar": "👨‍⚕️"}
    ]

for msg in st.session_state.messages:
    avatar = msg.get("avatar", "👨‍⚕️") if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 เอกสารอ้างอิงที่ใช้ตอบ"):
                for src in msg["sources"]:
                    st.write(f"• **{src['source']}**: {src['content']}")

if user_query := st.chat_input("พิมพ์คำถามเกี่ยวกับอาหารสุขภาพที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    search_results = search_docs(user_query, k=3)
    
    if not search_results:
        answer = "ไม่พบข้อมูลในคลังเอกสารอ้างอิง"
        retrieved_docs = []
    else:
        retrieved_docs = [item[0] for item in search_results]
        context_str = "\n\n".join([f"[แหล่งที่มา: {d['source']}]\n{d['content']}" for d in retrieved_docs])
        
        system_instruction = f"""คุณคือระบบผู้ช่วยตอบคำถามด้านโภชนาการและการเลือกอาหารเพื่อสุขภาพ ให้ตอบคำถามโดยอ้างอิงจาก Context ที่กำหนดให้เท่านั้น

ข้อบังคับเข้มงวด:
1. ให้ตอบคำถามให้อ่านง่าย ชัดเจน สรุปความเป็นข้อๆ ได้หากเหมาะสม
2. หากใน Context ไม่มีเนื้อหาที่ตรงกับคำถาม หรือข้อมูลไม่เพียงพอ ให้ตอบว่า "ไม่พบข้อมูล" เท่านั้น ห้ามคาดเดา หรือนำความรู้ภายนอกมาตอบโดยเด็ดขาด
3. อ้างอิงชื่อไฟล์แหล่งที่มาเสมอ

Context:
{context_str}
"""
        with st.chat_message("assistant"):
            with st.spinner("ระบบกำลังประมวลผลคำตอบ..."):
                try:
                    gemini_model = genai.GenerativeModel(
                        model_name="gemini-3.8-flash",
                        system_instruction=system_instruction
                    )
                    response = gemini_model.generate_content(user_query)
                    answer = response.text
                except Exception as e:
                    answer = f"เกิดข้อผิดพลาดในการประมวลผลของระบบ: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(answer)
        if retrieved_docs:
            with st.expander("📌 เอกสารอ้างอิงที่ใช้ตอบ"):
                for d in retrieved_docs:
                    st.write(f"• **{d['source']}**: {d['content']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_docs
    })