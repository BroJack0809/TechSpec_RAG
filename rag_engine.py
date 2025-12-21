import os
import sys
import shutil
import nest_asyncio

# LlamaIndex 核心組件
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

# Gemini 相關組件
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# LlamaParse 解析器
from llama_parse import LlamaParse

# BM25 關鍵字檢索
from llama_index.retrievers.bm25 import BM25Retriever

# 解決異步環境問題
nest_asyncio.apply()

# ================= 設定區 (請確認這裡) =================

# 1. API Keys (請自行填入或確保環境變數已設定)
# os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..." 
# os.environ["GOOGLE_API_KEY"] = "AIza..."


# 2. 檔案路徑 (使用原始字串 r"..." 避免 Windows 路徑錯誤)
PDF_PATH = "TSMC_2024 Annual Report-C.pdf"
PERSIST_DIR = "./storage"

# 3. [重要] 是否強制重新解析？
# 設為 True: 每次執行都會刪除舊索引，重新跑 LlamaParse (會扣額度，但確保資料最新)
# 設為 False: 如果有存檔就直接讀取 (省錢模式)
# 建議：第一次跑這份 PDF 時設為 True，跑完確認沒問題後改成 False
FORCE_RELOAD = True 

# ======================================================

def init_settings():
    """初始化全域模型設定"""
    
    # [修改] 使用您指定的 Gemini 2.5 Flash
    # 注意：如果遇到 429 Resource Exhausted，請改回 "models/gemini-1.5-flash"
    Settings.llm = Gemini(model="models/gemini-2.5-flash")
    
    # [維持] Embedding 模型必須使用專用的 Embedding 模型，不能用 Chat 模型
    Settings.embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.environ.get("GOOGLE_API_KEY")
    )
    
    # [優化] 加大 Chunk Size 以容納長表格
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200
    Settings.embed_batch_size = 10 

def get_index():
    """取得索引：處理存檔與重新解析的邏輯"""
    
    # 處理強制重跑邏輯
    if FORCE_RELOAD and os.path.exists(PERSIST_DIR):
        print(f"🧹 FORCE_RELOAD 為 True，正在刪除舊的索引資料夾 {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    # --- 情況 A: 讀取舊檔 ---
    if os.path.exists(PERSIST_DIR):
        print(f"📂 發現已存在的索引 ({PERSIST_DIR})，正在讀取... (省錢模式)")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index

    # --- 情況 B: 建立新索引 ---
    else:
        print("🚀 開始執行 LlamaParse 解析與向量化... (這會花一點時間)")
        
        # 設定 LlamaParse (包含針對年報的指令)
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
        
        # 檢查檔案是否存在
        if not os.path.exists(PDF_PATH):
            print(f"❌ 錯誤：找不到檔案，請檢查路徑：{PDF_PATH}")
            sys.exit(1)

        print(f"📄 正在讀取檔案：{PDF_PATH}")
        documents = SimpleDirectoryReader(
            input_files=[PDF_PATH],
            file_extractor=file_extractor
        ).load_data()
        
        # 印出解析預覽，確認表格有沒有被抓到
        print("\n--- LlamaParse 解析預覽 (隨機抽樣) ---")
        preview_text = documents[min(3, len(documents)-1)].text[:500] # 看第 4 頁或最後一頁
        print(preview_text)
        print("--------------------------------------\n")

        # 建立向量索引
        print("⚡ 正在建立 Vector Index (Chunk Size: 2048)...")
        index = VectorStoreIndex.from_documents(documents)
        
        # 存檔
        print(f"💾 正在將索引儲存到 {PERSIST_DIR}...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        return index

# 自定義混合檢索器 (Bypass LlamaIndex 的 FusionMode 檢查)
class CustomHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle):
        # 1. 取得向量檢索結果
        vec_nodes = self.vector_retriever.retrieve(query_bundle)
        # 2. 取得關鍵字檢索結果
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        # 3. 合併結果 (去重複)
        all_nodes = {}
        for node in vec_nodes:
            all_nodes[node.node.node_id] = node
        for node in bm25_nodes:
            if node.node.node_id not in all_nodes:
                all_nodes[node.node.node_id] = node
        
        # 轉回列表並回傳前 20 筆
        combined_list = list(all_nodes.values())
        return combined_list[:20] 

def create_hybrid_query_engine(index):
    """建立混合檢索查詢引擎"""
    print("🔧 正在初始化混合檢索器 (Custom Hybrid)...")
    
    # 向量檢索 (語意)
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10
    )

    # 關鍵字檢索 (精準)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=10
    )

    # 組合
    retriever = CustomHybridRetriever(vector_retriever, bm25_retriever)

    # 中文 Prompt
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

    # 建立引擎
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=chinese_qa_prompt
    )
    
    return query_engine

# ================= 主程式 =================

if __name__ == "__main__":
    try:
        # 1. 初始化設定
        init_settings()
        
        # 2. 取得索引
        index = get_index()
        
        # 3. 建立混合檢索引擎
        query_engine = create_hybrid_query_engine(index)

        print("\n==================================================")
        print(f"🤖 RAG 系統已啟動 (Model: Gemini 2.5 Flash)")
        print(f"📄 目標檔案: {os.path.basename(PDF_PATH)}")
        print("💡 提示：輸入 'q' 離開")
        print("==================================================\n")

        # 4. 互動迴圈
        while True:
            user_input = input("請輸入您的問題: ").strip()
            
            if user_input.lower() in ['q', 'exit', 'quit', '離開']:
                print("👋 掰掰！")
                break
            
            if not user_input:
                continue

            print("🤖 AI 正在思考中...")
            response = query_engine.query(user_input)
            
            print(f"\n💬 回答:\n{response}")
            
            # 來源追蹤
            print("\n🕵️ [來源追蹤] 參考了以下片段：")
            for node in response.source_nodes:
                score = f"{node.score:.2f}" if node.score is not None else "Hybrid"
                # 預覽內容 (移除換行方便閱讀)
                preview = node.node.get_text()[:60].replace('\n', ' ')
                print(f"   - [分數 {score}] {preview}...")
            print("-" * 50)

    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()