# -*- coding: utf-8 -*-
"""
RAG 核心引擎與 ETL 流程 (RAG Core Engine & ETL Pipeline)

此模組負責處理 RAG 系統的後端邏輯，包含：
1. 資料擷取 (Data Ingestion): 使用 LlamaParse 解析 PDF 年報。
2. 索引建構 (Indexing): 建立向量索引 (Vector Index) 並持久化儲存。
3. 查詢引擎工廠 (Query Engine Factory): 封裝混合檢索 (Hybrid Search) 的初始化邏輯。

用途:
- 作為 `main` 執行時，執行 ETL 流程並生成索引。
- 作為模組被 `app.py` 匯入時，提供查詢引擎建構功能。
"""

import os
import sys
import shutil
import nest_asyncio

# --- LlamaIndex 核心組件 (Core Components) ---
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage, 
    Settings, 
    PromptTemplate
)
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

# --- Google Gemini 模型整合 (Check Gemini Integration) ---
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- 資料解析器 (Advanced Parsing) ---
from llama_parse import LlamaParse

# --- 關鍵字檢索 (Keyword Search) ---
from llama_index.retrievers.bm25 import BM25Retriever

# 解決異步環境下的 Event Loop 問題
nest_asyncio.apply()

# ================= 全域配置 (Global Configuration) =================

# 1. API Keys (建議使用環境變數管理，避免 Hardcode)
# os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..." 
# os.environ["GOOGLE_API_KEY"] = "AIza..."

# 2. 檔案路徑配置 (File Path Configuration)
# 使用原始字串 (Raw String) 防止 Windows 路徑轉義問題
PDF_PATH = "TSMC_2024 Annual Report-C.pdf"
PERSIST_DIR = "./storage"

# 3. 索引重建策略 (Re-indexing Strategy)
# FORCE_RELOAD = True:  強制清除舊索引，重新執行 ETL (消耗 LlamaCloud 額度)
# FORCE_RELOAD = False: 若索引存在則直接讀取 (Cache Hit)，節省成本與時間
# 建議開發階段設為 True，部署後設為 False
FORCE_RELOAD = True 

# =================================================================

def init_settings():
    """
    初始化 LlamaIndex 全域設定 (Global Settings Initialization)。
    
    配置預設的 LLM 與 Embedding 模型，以及 Chunking 策略。
    此函數應在系統啟動時最先被調用。
    """
    
    # [Model Configuration] 使用 Google Gemini 2.5 Flash
    # 優勢：速度快、Context Window 大的輕量級模型
    # 注意：若遇 Rate Limit (429)，可降級至 "models/gemini-1.5-flash"
    Settings.llm = Gemini(model="models/gemini-2.5-flash")
    
    # [Embedding Configuration] 使用多語言文字嵌入模型
    # 必須指定 api_key，以確保與 LLM 分開計費或管理
    Settings.embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.environ.get("GOOGLE_API_KEY")
    )
    
    # [Chunking Strategy] 針對年報長文本優化
    # chunk_size: 2048 Tokens (涵蓋更完整的上下文，如長表格)
    # chunk_overlap: 200 Tokens (保持語意連貫性)
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200
    Settings.embed_batch_size = 10 

def get_index():
    """
    獲取向量索引 (Index Retrieval Strategy)。
    
    實現「讀寫分離」邏輯：
    1. 若 FORCE_RELOAD 為 True，則刪除舊索引。
    2. 若索引存在，直接從磁碟載入 (Load from Disk)。
    3. 若索引不存在，執行完整 ETL 流程：解析 -> 向量化 -> 儲存。
    
    Returns:
        VectorStoreIndex: 初始化完成的向量索引物件。
    """
    
    # 處理強制重跑邏輯 (Force Reload Logic)
    if FORCE_RELOAD and os.path.exists(PERSIST_DIR):
        print(f"🧹 [System] FORCE_RELOAD=True，清除舊索引目錄：{PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    # --- 策略 A: 快取命中 (Cache Hit) ---
    if os.path.exists(PERSIST_DIR):
        print(f"📂 [Storage] 發現現有索引 ({PERSIST_DIR})，直接載入...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index

    # --- 策略 B: 冷啟動 (Cold Start / ETL) ---
    else:
        print("🚀 [ETL] 開始執行 LlamaParse 解析與向量化流程...")
        
        # 配置 LlamaParse (Advanced PDF Parsing)
        # 針對繁體中文年報優化，提供特定的 Prompt 處理跨頁表格
        parser = LlamaParse(
            result_type="markdown",
            verbose=True,
            language="ch_tra",
            parsing_instruction="""
            這是一份年報。請將其解析為標準 Markdown。
            重要規則：
            1. 遇到跨頁的表格（如董事會成員名單、財務報表），請盡量將其合併為一個完整的 Markdown 表格。
            2. 絕對不要遺漏表格中的任何一列（Row）數據或數字。
            3. 保留封面上的關鍵資訊（股票代號、刊印日期）。
            """
        )
        
        file_extractor = {".pdf": parser}
        
        # 檔案完整性檢查
        if not os.path.exists(PDF_PATH):
            print(f"❌ [Error] 找不到目標檔案：{PDF_PATH}")
            sys.exit(1)

        print(f"📄 [Ingestion] 讀取檔案：{PDF_PATH}")
        documents = SimpleDirectoryReader(
            input_files=[PDF_PATH],
            file_extractor=file_extractor
        ).load_data()
        
        # 解析結果抽樣檢查 (Sanity Check)
        print("\n--- LlamaParse 解析預覽 (Sampling) ---")
        preview_text = documents[min(3, len(documents)-1)].text[:500] 
        print(preview_text)
        print("--------------------------------------\n")

        # 建立向量索引 (Indexing)
        print("⚡ [Vector Store] 正在建立 Vector Index (Chunk Size: 2048)...")
        index = VectorStoreIndex.from_documents(documents)
        
        # 持久化儲存 (Persistence)
        print(f"💾 [Storage] 儲存索引至 {PERSIST_DIR}...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        return index

# 自定義混合檢索器 (Custom Hybrid Retriever)
# 繼承 BaseRetriever 以整合至 LlamaIndex 流程
class CustomHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle):
        # 1. 執行向量檢索 (Vector Search)
        vec_nodes = self.vector_retriever.retrieve(query_bundle)
        # 2. 執行關鍵字檢索 (BM25 Search)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        # 3. 結果融合 (Result Fusion)
        # 使用 Map 結構去重，優先保留向量檢索的分數與節點
        all_nodes = {}
        for node in vec_nodes:
            all_nodes[node.node.node_id] = node
        for node in bm25_nodes:
            if node.node.node_id not in all_nodes:
                all_nodes[node.node.node_id] = node
        
        # 回傳融合後的 Top-20 結果
        combined_list = list(all_nodes.values())
        return combined_list[:20] 

def create_hybrid_query_engine(index):
    """
    工廠模式 (Factory Pattern) 建立混合檢索查詢引擎。
    
    Args:
        index (VectorStoreIndex): 已載入的向量索引。
        
    Returns:
        RetrieverQueryEngine: 配置完成的查詢引擎。
    """
    print("🔧 [Factory] 初始化混合檢索器 (Custom Hybrid)...")
    
    # 實作檢索器元件
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10
    )

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=10
    )

    # 組合
    retriever = CustomHybridRetriever(vector_retriever, bm25_retriever)

    # 定義系統提示詞 (System Prompt)
    # 強調來源依據與回應語言
    qa_prompt_str = (
        "以下是參考文件內容：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "請僅根據上述參考文件內容，回答使用者的問題: {query_str}\n"
        "嚴格禁止編造文件中未提及的人名、數字或職稱。\n"
        "如果參考文件中沒有完整的名單或數據，請回答「文件中僅提及部分內容」，並列出你有看到的即可。\n"
        "請務必使用「繁體中文」回答。\n"
    )
    chinese_qa_prompt = PromptTemplate(qa_prompt_str)

    # 建構引擎
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=chinese_qa_prompt
    )
    
    return query_engine

# ================= 程式入口 (Modules Entry Point) =================

if __name__ == "__main__":
    try:
        # 1. 環境初始化 (Bootstrap)
        init_settings()
        
        # 2. 準備索引 (Prepare Index)
        index = get_index()
        
        # 3. 建立引擎 (Build Engine)
        query_engine = create_hybrid_query_engine(index)

        print("\n==================================================")
        print(f"🤖 RAG 系統已啟動 (Model: Gemini 2.5 Flash)")
        print(f"📄 目標檔案: {os.path.basename(PDF_PATH)}")
        print("💡 提示：輸入 'q' 離開")
        print("==================================================\n")

        # 4. 互動式 REPL 迴圈 (Interactive Loop)
        while True:
            user_input = input("請輸入您的問題: ").strip()
            
            if user_input.lower() in ['q', 'exit', 'quit', '離開']:
                print("👋 程式結束 (Terminated)")
                break
            
            if not user_input:
                continue

            print("🤖 AI 正在推論中 (Inference)...")
            response = query_engine.query(user_input)
            
            print(f"\n💬 回答:\n{response}")
            
            # 來源可解釋性 (Explainability)
            print("\n🕵️ [來源追蹤] 參考了以下片段：")
            for node in response.source_nodes:
                score = f"{node.score:.2f}" if node.score is not None else "Hybrid"
                # 預覽內容
                preview = node.node.get_text()[:60].replace('\n', ' ')
                print(f"   - [分數 {score}] {preview}...")
            print("-" * 50)

    except Exception as e:
        print(f"\n❌ 發生未預期的錯誤 (Unexpected Error): {e}")
        import traceback
        traceback.print_exc()