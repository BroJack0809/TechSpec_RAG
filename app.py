# -*- coding: utf-8 -*-
"""
應用程式入口點 (Entry Point)
此模組使用 Streamlit 構建 RAG (Retrieval-Augmented Generation) 系統的前端介面。
它整合了 Google Gemini 模型與混合檢索策略 (Hybrid Search)，提供針對台積電年報的問答服務。

主要功能:
1. 提供使用者介面，用於輸入 Google API Key 與查詢問題。
2. 實作 Singleton 模式載入 RAG 引擎，避免重複初始化資源。
3. 展示檢索來源 (Source Nodes) 與信心分數 (Confidence Score)。
"""

import streamlit as st
import os
import nest_asyncio
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- Import 路徑修正與模組依賴 ---
# 引入核心檢索器介面與實作
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate

# 解決異步 (AsyncIO) 在 Streamlit 環境下的事件迴圈衝突問題
nest_asyncio.apply()

# --- Streamlit 頁面組態設定 ---
st.set_page_config(page_title="台積電年報 AI 助手", layout="wide")
st.title("🤖 TSMC 年報 RAG 助手 (Gemini 2.5 + Hybrid Search)")

# --- 側邊欄配置 (Sidebar Configuration) ---
# 負責處理環境變數與 API Key 的輸入，確保應用程式安全性
with st.sidebar:
    st.header("⚙️ 系統設定")
    
    # 嘗試預先讀取環境變數，以提升開發者體驗 (DX)
    default_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input("Google API Key", value=default_key, type="password")
    
    # 若使用者有輸入 Key，則更新環境變數
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    st.info("系統提示：請確保 ./storage 持久化目錄已建立 (需預先執行 Data Ingestion Pipeline)。")
    if st.button("🔄 重新載入應用程式"):
        st.rerun()

# --- 先決條件檢查 (Pre-flight Check) ---
# 強制要求 API Key 存在，否則阻斷執行流程 (Circuit Breaker)
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⬅️ 請先於側邊欄設定 Google API Key 以初始化 LLM 服務。")
    st.stop()
# --------------------

# --- 核心類別定義 (Core Definitions) ---

class CustomHybridRetriever(BaseRetriever):
    """
    自定義混合檢索器 (Custom Hybrid Retriever)
    
    實作 RAG 的混合檢索策略，結合向量檢索 (Vector Search) 與關鍵字檢索 (BM25)。
    
    Attributes:
        vector_retriever (VectorIndexRetriever): 負責語意相似度檢索 (Semantic Similarity)。
        bm25_retriever (BM25Retriever): 負責關鍵字精準匹配 (Exact Keyword Match)。
    """
    
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle):
        """
        執行檢索邏輯，並合併(Merge)與去重(Deduplicate)兩種檢索器的結果。
        
        Args:
            query_bundle (QueryBundle): 包含查詢字串與相關資訊的物件。
            
        Returns:
            List[NodeWithScore]: 合併後的檢索節點列表。
        """
        try:
            # 1. 平行執行兩種檢索策略
            vec_nodes = self.vector_retriever.retrieve(query_bundle)
            bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
            
            # 2. 結果合併策略 (Merge Strategy)
            # 使用 Dictionary 以 node_id 為鍵進行去重，優先保留向量檢索結果
            all_nodes = {}
            for node in vec_nodes: 
                all_nodes[node.node.node_id] = node
            for node in bm25_nodes:
                if node.node.node_id not in all_nodes: 
                    all_nodes[node.node.node_id] = node
            
            # 3. 回傳前 20 筆最相關的結果 (Top-K)
            return list(all_nodes.values())[:20]
        except Exception as e:
            # 錯誤處理：記錄錯誤並回傳空列表以避免 Crash
            print(f"Retrieval Error: {e}")
            return []

# --- 依賴注入與資源初始化 (Dependency Injection & Initialization) ---

@st.cache_resource
def load_rag_engine():
    """
    初始化並載入 RAG 查詢引擎 (Query Engine)。
    
    使用 @st.cache_resource 裝飾器實現 Singleton 模式，
    確保在 Streamlit 的多次互動中，模型與索引只會被載入一次，優化效能。
    
    Returns:
        RetrieverQueryEngine: 初始化完成的查詢引擎實例，若失敗則回傳 None。
    """
    persist_dir = "./storage"
    
    # 檢查持久化儲存層是否存在
    if not os.path.exists(persist_dir):
        return None
    
    try:
        # 設定 LLM 與 Embedding 模型 (Global Settings Configuration)
        Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"])
        Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=os.environ["GOOGLE_API_KEY"])
        
        # 從磁碟載入索引結構 (Load Index from Disk)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        
        # 實例化檢索器元件 (Instantiate Retrievers)
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
        bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
        
        # 依賴注入：組合混合檢索器
        retriever = CustomHybridRetriever(vector_retriever, bm25_retriever)
        
        # 定義系統提示詞 (System Prompt Construction)
        # 強調「基於事實」(Grounding) 與繁體中文輸出的規範
        qa_prompt_str = (
            "以下是參考文件內容：\n---------------------\n{context_str}\n---------------------\n"
            "請僅根據上述參考文件內容，回答使用者的問題: {query_str}\n"
            "嚴格禁止編造文件中未提及的人名、數字或職稱 (Hallucination Prevention)。\n"
            "請務必使用「繁體中文」回答，若是表格數據請排版整齊。\n"
        )
        
        # 構建最終查詢引擎
        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=PromptTemplate(qa_prompt_str)
        )
    except Exception as e:
        st.error(f"引擎初始化失敗 (Initialization Failed): {e}")
        return None

# --- 主應用程式邏輯 (Main Application Logic) ---

# 初始化對話歷史狀態 (Session State Management)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染歷史對話訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 載入核心引擎
engine = load_rag_engine()

if engine is None:
    st.error("❌ 系統錯誤：找不到 ./storage 索引目錄。請確保已執行 ETL Pipeline (`python rag_engine.py`) 生成索引。")
else:
    # 處理使用者輸入事件
    if prompt := st.chat_input("請輸入問題 (例如：董事會成員有哪些？)"):
        # 更新 UI 並記錄使用者訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 觸發 AI 回應流程
        with st.chat_message("assistant"):
            with st.spinner("AI 正在進行語意檢索與生成 (RAG Processing)..."):
                try:
                    # 執行查詢
                    response = engine.query(prompt)
                    st.markdown(response.response)
                    
                    # 使用 Expander 展示可解釋性資訊 (Explainability)
                    with st.expander("🕵️ 參考來源片段 (Source Context)"):
                        for node in response.source_nodes:
                            score = f"{node.score:.2f}" if node.score is not None else "Hybrid"
                            st.caption(f"**[關聯分數 {score}]**")
                            st.text(node.node.get_text()[:200] + "...")
                            st.divider()

                    # 記錄 AI 回應至 Session State
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                
                except Exception as e:
                    st.error(f"執行階段錯誤 (Runtime Error): {e}")