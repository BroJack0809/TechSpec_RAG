# TSMC Annual Report RAG 專案

本專案使用 RAG (Retrieval-Augmented Generation) 技術，針對台積電民國 113 年度年報 (2024 Annual Report) 進行分析與問答。

## 🛠️ 使用技術

本專案使用了以下關鍵技術與工具：

*   **LlamaIndex**: 構建 RAG 應用程式的核心框架。
*   **Gemini 2.5 Flash**: Google 的高效能大型語言模型，用於生成回答。
*   **Gemini Embeddings (text-embedding-004)**: 用於將文字轉化為向量。
*   **LlamaParse**: 專門用於解析 PDF 的工具，能有效處理複雜的表格與文件結構。
*   **BM25**: 用於關鍵字檢索，與向量檢索結合形成混合檢索 (Hybrid Search)。
*   **Streamlit**: 用於建立使用者互動介面。

## 📦 安裝說明

在使用本專案之前，請確保您已安裝 Python 環境。

1.  **Clone 專案**
    ```bash
    git clone https://github.com/BroJack0809/TechSpec_RAG.git
    cd TechSpec_RAG
    ```

2.  **安裝必要套件**
    請執行以下指令安裝所有相依套件：
    ```bash
    pip install -r requirements.txt
    ```

3.  **環境變數設定**
    本專案需要 Google API KEY。您可以在執行時輸入，或設定系統環境變數：
    *   `GOOGLE_API_KEY`: 您的 Gemini API Key
    *   `LLAMA_CLOUD_API_KEY`: LlamaParse 需要的 API Key (若要執行 `rag_engine.py` 重新解析 PDF)

## 🚀 使用說明

本專案包含兩個主要執行檔：`rag_engine.py` (後端索引建立) 和 `app.py` (前端介面)。

### 步驟 1: 建立或更新索引 (`rag_engine.py`)

如果您是**第一次使用**，或想要**分析新的 PDF 檔案**，請先執行此步驟。

> **注意**: 如果您只是想執行現有的 RAG 查詢，且 `storage` 資料夾已存在，可以跳過此步驟。

**若需分析新檔案或重新分析：**
1.  請先刪除專案目錄下的 `storage` 資料夾。
2.  確保 PDF 檔案 (`TSMC_2024 Annual Report-C.pdf`) 位於專案根目錄。
3.  執行以下指令：
    ```bash
    python rag_engine.py
    ```
    *   此步驟會呼叫 LlamaParse 解析 PDF，建立向量索引，並儲存至 `storage` 資料夾。
    *   執行完成後，您可以在終端機中簡單測試問答。

### 步驟 2: 啟動 Web 應用程式 (`app.py`)

建立好索引 (`storage` 資料夾) 後，即可啟動圖形化介面進行操作。

1.  執行 Streamlit 應用程式：
    ```bash
    streamlit run app.py
    ```
2.  瀏覽器將自動開啟應用程式 (預設網址為 `http://localhost:8501`)。
3.  在側邊欄輸入您的 Google API Key (若未設定環境變數)。
4.  開始輸入問題進行對話！

## 📂 檔案結構

*   `app.py`: Streamlit 前端應用程式主程式。
*   `rag_engine.py`: 負責文件解析、索引建立與 RAG 核心邏輯。
*   `requirements.txt`: 專案相依套件列表。
*   `TSMC_2024 Annual Report-C.pdf`: 預設分析的年報檔案。
*   `storage/`: (自動產生) 存放向量索引的資料夾。
