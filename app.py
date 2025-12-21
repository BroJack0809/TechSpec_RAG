# -*- coding: utf-8 -*-
import streamlit as st
import os
import nest_asyncio
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- Import 路徑修正 ---
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
# ---------------------

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate

# 解決異步問題
nest_asyncio.apply()

# --- 設定頁面 ---
st.set_page_config(page_title="台積電年報 AI 助手", layout="wide")
st.title("🤖 TSMC 年報 RAG 助手 (Gemini 2.5 + Hybrid Search)")

# --- 側邊欄設定 (關鍵修正：加入紅綠燈) ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    # 嘗試從環境變數讀取預設值
    default_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input("Google API Key", value=default_key, type="password")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    st.info("請確保 ./storage 資料夾已經建立完畢 (請先執行過一次 main.py)")
    if st.button("🔄 重新整理"):
        st.rerun()

# --- [紅綠燈檢查點] ---
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⬅️ 請先在側邊欄輸入 Google API Key 才能開始對話！")
    st.stop()  # <--- 這裡會暫停程式執行，直到有 Key 為止
# --------------------

# --- 核心類別定義 ---
class CustomHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle):
        try:
            vec_nodes = self.vector_retriever.retrieve(query_bundle)
            bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
            
            all_nodes = {}
            for node in vec_nodes: 
                all_nodes[node.node.node_id] = node
            for node in bm25_nodes:
                if node.node.node_id not in all_nodes: 
                    all_nodes[node.node.node_id] = node
            
            return list(all_nodes.values())[:20]
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return []

# --- 初始化引擎 ---
@st.cache_resource
def load_rag_engine():
    persist_dir = "./storage"
    
    if not os.path.exists(persist_dir):
        return None
    
    try:
        # 設定模型 (確保這裡執行時已經有 API Key 了)
        Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"])
        Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=os.environ["GOOGLE_API_KEY"])
        
        # 載入索引
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        
        # 建立檢索器
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
        bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
        
        # 組合混合檢索器
        retriever = CustomHybridRetriever(vector_retriever, bm25_retriever)
        
        # Prompt
        qa_prompt_str = (
            "以下是參考文件內容：\n---------------------\n{context_str}\n---------------------\n"
            "請僅根據上述參考文件內容，回答使用者的問題: {query_str}\n"
            "嚴格禁止編造文件中未提及的人名、數字或職稱。\n"
            "請務必使用「繁體中文」回答，若是表格數據請排版整齊。\n"
        )
        
        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=PromptTemplate(qa_prompt_str)
        )
    except Exception as e:
        st.error(f"引擎載入失敗: {e}")
        return None

# --- 主邏輯 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 載入引擎
engine = load_rag_engine()

if engine is None:
    st.error("❌ 找不到 ./storage 索引資料夾！請先執行 `python main.py` 來生成索引。")
else:
    # 接收使用者輸入
    if prompt := st.chat_input("請輸入問題 (例如：董事會成員有哪些？)"):
        # 顯示使用者訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 回答
        with st.chat_message("assistant"):
            with st.spinner("AI 正在翻閱年報中..."):
                try:
                    response = engine.query(prompt)
                    st.markdown(response.response)
                    
                    # 顯示來源
                    with st.expander("🕵️ 參考來源片段"):
                        for node in response.source_nodes:
                            score = f"{node.score:.2f}" if node.score is not None else "Hybrid"
                            st.caption(f"**[分數 {score}]**")
                            st.text(node.node.get_text()[:200] + "...")
                            st.divider()

                    # 儲存 AI 回答
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                
                except Exception as e:
                    st.error(f"發生錯誤: {e}")