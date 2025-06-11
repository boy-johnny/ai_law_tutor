# app.py
import streamlit as st
import json
from script.main import GeminiRAG  # 從你的 script 資料夾導入 GeminiRAG 類別
import os
import dotenv
dotenv.load_dotenv()

# --- 1. 頁面配置與標題 ---
st.set_page_config(page_title="行政法 AI 導師", page_icon="⚖️", layout="wide")
st.title("⚖️ 行政法 AI 導師")
st.markdown("你好！我是一位基於行政法資料的 AI 導師。你可以問我問題、要求總結，或讓我為你出幾道測驗題。")

# --- 2. 載入核心 RAG 系統 (最重要的一步) ---
# @st.cache_resource 是 Streamlit 的魔法指令，它會快取這個函式的回傳結果。
# 這意味著 GeminiRAG 系統只會在第一次啟動時被完整載入一次（耗時的過程），
# 之後每次使用者互動，都會直接使用快取好的物件，讓應用反應極速。
@st.cache_resource
def load_rag_system():
    """載入並快取 RAG 系統，避免每次互動都重新載入"""
    api_key = os.getenv("GOOGLE_API_KEY") # 確保你的 API KEY 在環境變數中
    if not api_key:
        st.error("請設置 GOOGLE_API_KEY 環境變數！")
        return None
    
    # 注意路徑，確保它們是正確的
    rag_system = GeminiRAG(
        api_key=api_key, 
        docs_dir="my_textbook_data", 
        db_path="my_textbook.pkl"
    )
    return rag_system

# 執行載入函式
rag_system = load_rag_system()

if rag_system is None:
    st.stop() # 如果系統載入失敗，則停止執行
# --- 創建多任務介面 ---
tab1, tab2 = st.tabs(["⚖️ 法律概念提問", "✍️ 申論題寫作練習"])

# --- 介面 1: 法律概念提問 ---
with tab1:
    st.header("對任何行政法概念提出疑問")
    concept_query = st.text_input(
        "在這裡輸入你的問題...", 
        placeholder="例如：什麼是行政處分？信賴保護原則的要件是什麼？",
        key="concept_input"
    )

    if st.button("開始提問", key="concept_button"):
        if concept_query:
            with st.spinner("老師正在思考中..."):
                # <<< 變更點：呼叫時傳入 task_type >>>
                response = rag_system.process_query(concept_query, task_type="explain_concept")

            st.subheader("老師的回答：")
            if "error" in response:
                st.error(f"處理時發生錯誤：{response['error']}")
            elif "explanation" in response:
                st.info(response.get("explanation", "無內容"))
                st.caption(f"關鍵法條：{', '.join(response.get('key_statutes', ['未提供']))}")
                st.caption(f"引用來源：{', '.join(response.get('source_citations', ['未提供']))}")
            else:
                st.warning("收到的回答格式無法識別，顯示原始 JSON 結果：")
                st.json(response)

# --- 介面 2: 申論題寫作練習 ---
with tab2:
    st.header("練習回答申論題並獲得評語")
    
    # 這裡我們先用 text_input，未來可以改成從資料庫讀取考題的 selectbox
    essay_question = st.text_input(
        "請貼上或輸入申論題的題目",
        placeholder="例如：何謂「行政自我拘束原則」？其理論基礎及具體適用之要件為何？請申論之。",
        key="essay_question"
    )
    
    user_answer = st.text_area(
        "在這裡輸入你的答案",
        height=300,
        key="essay_answer"
    )

    if st.button("批改我的答案", key="essay_button"):
        if essay_question and user_answer:
            with st.spinner("老師正在批改你的申論題..."):
                 # <<< 變更點：呼叫時傳入 query, task_type 和 user_answer >>>
                response = rag_system.process_query(
                    query=essay_question, 
                    task_type="grade_essay", 
                    user_answer=user_answer
                )
            
            st.subheader("老師的評語：")
            if "error" in response:
                st.error(f"處理時發生錯誤：{response['error']}")
            elif "strengths" in response:
                st.success("**做得好的地方：**")
                st.markdown(response.get("strengths", "無"))
                
                st.warning("**可以改進的地方：**")
                st.markdown(response.get("areas_for_improvement", "無"))

                st.info("**總體評價：**")
                st.markdown(response.get("overall_comment", "無"))
            else:
                st.warning("收到的回答格式無法識別，顯示原始 JSON 結果：")
                st.json(response)
        else:
            st.warning("請務必輸入題目和你的答案。")
