# script/rag.py
import os
import pickle
import json # <<< 變更點：導入 json 模組
from typing import List, Dict
import numpy as np
from google.generativeai import GenerativeModel, configure, embedding
from dotenv import load_dotenv
import logging
from functools import lru_cache
import glob
import time
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiRAG:
    # __init__, _load_vector_db_from_file, _save_vector_db_to_file,
    # _load_documents, _split_text, _initialize_vector_db, _retrieve_documents
    # 等方法保持不變，此處省略以保持簡潔。
    # 請確保你保留了那些方法的完整程式碼。
    def __init__(self, api_key: str, docs_dir: str, db_path: str = "vector_db.pkl"):
        """
        初始化 Gemini RAG 系統
        Args:
            api_key (str): Google AI Studio API key
            docs_dir (str): 文檔目錄路徑
            db_path (str): 向量資料庫存檔路徑
        """
        configure(api_key=api_key)
        self.model = GenerativeModel("gemini-1.5-pro")
        self.query_count = 0
        self.total_response_time = 0.0
        self.db_path = db_path # 儲存檔案路徑

        if os.path.exists(self.db_path):
            self._load_vector_db_from_file()
        else:
            logger.info("未找到本地向量資料庫檔案，開始建立新的資料庫...")
            self.docs_dir = docs_dir
            self.documents = self._load_documents()
            if self.documents:
                self._initialize_vector_db()
                self._save_vector_db_to_file()
            else:
                logger.warning("沒有載入任何文檔，向量資料庫為空。")
                self.vector_db = {}
        
        # <<< 變更點：lru_cache 現在會快取字典結果
        self.process_query = lru_cache(maxsize=128)(self._process_query_internal)
        logger.info("GeminiRAG 系統初始化完成")

    def _load_vector_db_from_file(self):
        """從 pickle 檔案載入向量資料庫"""
        logger.info(f"從 '{self.db_path}' 載入已存在的向量資料庫...")
        try:
            with open(self.db_path, 'rb') as f:
                self.vector_db = pickle.load(f)
            logger.info(f"向量資料庫載入成功，包含 {len(self.vector_db)} 個文檔片段。")
        except Exception as e:
            logger.error(f"從檔案載入向量資料庫失敗: {e}")
            raise

    def _save_vector_db_to_file(self):
        """將當前的向量資料庫儲存為 pickle 檔案"""
        logger.info(f"正在將新的向量資料庫儲存至 '{self.db_path}'...")
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.vector_db, f)
            logger.info("儲存成功。")
        except Exception as e:
            logger.error(f"儲存向量資料庫至檔案時失敗: {e}")
    def _load_documents(self):
        """
        從文檔目錄載入所有文檔，並驗證格式。
        """
        logger.info(f"從 '{self.docs_dir}' 目錄載入文檔...")
        docs = []
        all_files = glob.glob(os.path.join(self.docs_dir, "**/*"), recursive=True)

        for file_path in all_files:
            if os.path.isdir(file_path):
                continue

            file_name = os.path.basename(file_path)
            try:
                # --- 文字檔案處理 ---
                if file_name.endswith(('.md', '.txt')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        docs.append({"content": f.read(), "source": file_name})
                
                # --- JSONL 檔案處理 ---
                elif file_name.endswith('.jsonl'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            try:
                                data = json.loads(line)
                                # <<< 關鍵檢查點 >>>
                                if 'content' in data and 'source' in data:
                                    docs.append(data)
                                else:
                                    logger.warning(f"檔案 {file_name} 的第 {i+1} 行缺少 'content' 或 'source' 鍵，已略過。")
                            except json.JSONDecodeError:
                                logger.warning(f"無法解析檔案 {file_name} 的第 {i+1} 行，已略過。")

                # --- JSON 檔案處理 ---
                elif file_name.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_data = json.load(f)
                        # 處理整個檔案是一個文檔列表的情況
                        if isinstance(all_data, list):
                            for i, item in enumerate(all_data):
                                # <<< 關鍵檢查點 >>>
                                if isinstance(item, dict) and 'content' in item and 'source' in item:
                                    docs.append(item)
                                else:
                                    logger.warning(f"檔案 {file_name} 列表中的第 {i+1} 個項目格式不正確，已略過。")
                        # 處理整個檔案是一個單一文檔的情況
                        elif isinstance(all_data, dict):
                            # <<< 關鍵檢查點 >>>
                            if 'content' in all_data and 'source' in all_data:
                                docs.append(all_data)
                            else:
                                logger.warning(f"檔案 {file_name} 格式不正確，已略過。")

            except Exception as e:
                logger.error(f"讀取文件 {file_path} 時發生嚴重錯誤: {e}")

        logger.info(f"成功載入並驗證了 {len(docs)} 個文檔。")
        return docs
    
    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        將文本分割成小塊
        
        Args:
            text (str): 要分割的文本
            chunk_size (int): 每個塊的大小
            overlap (int): 重疊部分的大小
            
        Returns:
            List[str]: 文本塊列表
        """
        # 另一種可選策略：按段落分割
        # chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        # return chunks

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start += (chunk_size - overlap)
        return chunks
    
    def _initialize_vector_db(self):
        """初始化向量資料庫"""
        try:
            # 為每個文檔生成嵌入向量
            self.vector_db = {}
            # 使用 list comprehension 和 dict constructor 提高效率
            contents = [doc["content"] for doc in self.documents]
            embedding_results = embedding.embed_content(
                model="models/embedding-001",
                content=contents, # <<< 變更點：一次性傳送多個內容進行批次嵌入，速度更快
                task_type="retrieval_document"
            )
            embeddings = embedding_results['embedding']

            for i, doc in enumerate(self.documents):
                # 使用 content 作為 key 可能會因太長或重複而不穩定，使用索引更佳
                self.vector_db[i] = {
                    "content": doc["content"],
                    "embedding": embeddings[i],
                    "source": doc["source"]
                }
            logger.info(f"向量資料庫初始化完成，包含 {len(self.vector_db)} 個文檔")
        except Exception as e:
            logger.error(f"向量資料庫初始化失敗: {str(e)}")
            raise
    
    def _retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        檢索相關文檔
        
        Args:
            query (str): 查詢文字
            top_k (int): 返回的文檔數量
            
        Returns:
            List[Dict]: 相關文檔列表
        """
        if not self.vector_db:
             return []
        try:
            # 使用新的 embedding API
            query_embedding_result = embedding.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = query_embedding_result['embedding']
            
            # 計算相似度並排序
            query_embedding = np.array(query_embedding).reshape(1, -1)  # 調整 query_embedding 的形狀
            similarities = []
            for idx, data in self.vector_db.items():
                doc_embedding = np.array(data["embedding"]).reshape(1, -1)  # 調整 doc_embedding 的形狀
                similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]  # 計算 cosine 相似度
                similarities.append({
                    "content": data["content"],
                    "score": similarity,
                    "source": data["source"]
                })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"文檔檢索失敗: {str(e)}")
            return []


        # <<< 變更點 4：重構 _build_prompt 以處理多任務 >>>
    def _build_prompt(self, query: str, relevant_docs: List[Dict], task_type: str, user_answer: str = None) -> str:
        """根據查詢內容和任務類型，動態構建適合的提示"""
        context = "\n\n".join([
            f"來源: {doc.get('source', '未知來源')}\n內容: {doc.get('content', '')}"
            for doc in relevant_docs
        ]) if relevant_docs else "沒有找到相關的上下文資訊。"

        base_instructions = f"""
        # 角色
        你是一位資深且耐心的台灣行政法老師。你的知識庫僅限於我提供的法規、考古題和講義。

        # 核心規則
        1.  絕對禁止使用外部知識。
        2.  所有解釋和分析都必須引用具體的法條或講義來源。
        3.  使用台灣的法律用語。
        
        # 上下文資料
        {context}
        ---
        """

        task_instructions = ""
        # --- 根據任務類型選擇不同的指令 ---
        if task_type == "explain_concept":
            task_instructions = f"""
            # 任務：解析法律概念
            以清晰、有條理的方式，深入淺出地解釋以下問題。

            # 問題
            {query}

            # 輸出格式 (嚴格遵守此 JSON 結構)
            ```json
            {{
              "explanation": "對問題的詳細、完整的回答。",
              "key_statutes": ["列出回答中引用的最關鍵法條，例如：行政程序法 第 92 條"],
              "source_citations": ["參考的所有來源檔案列表"]
            }}
            ```
            """
        
        elif task_type == "grade_essay":
            # 為了能執行，我們先做一個簡化版的評分提示
            task_instructions = f"""
            # 任務：批改申論題 (簡化版)
            根據上下文資料，對學生的答案進行分析。

            # 考試題目
            {query}
            
            # 學生答案
            {user_answer}

            # 你的要求
            1. 判斷學生答案的優點。
            2. 指出學生答案可以改進的地方。
            3. 給予一個初步的總體評價。

            # 輸出格式 (嚴格遵守此 JSON 結構)
            ```json
            {{
              "strengths": "條列式說明學生答案的優點。",
              "areas_for_improvement": "條列式說明學生答案可以加強的地方。",
              "overall_comment": "給予一段總結性的評語。"
            }}
            ```
            """
        
        return base_instructions + task_instructions

         # <<< 變更點 1：修改 process_query 的簽名 >>>
    # 我們將快取從 process_query 移至 _process_query_internal，以處理更複雜的參數
    def process_query(self, query: str, task_type: str = "explain_concept", user_answer: str = None) -> Dict:
        """
        公開的查詢處理方法。
        Args:
            query (str): 主要查詢內容 (例如問題或考題題目)。
            task_type (str): 任務類型 ('explain_concept', 'grade_essay')。
            user_answer (str, optional): 使用者的申論題答案。
        Returns:
            Dict: 處理後的結果。
        """
        # 將所有需要快取的參數傳遞給內部函數
        # 我們將 query, task_type, user_answer 組成一個元組作為快取鍵
        return self._process_query_internal(query, task_type, user_answer)

    # <<< 變更點 2：修改內部函數和快取 >>>
    @lru_cache(maxsize=128)
    def _process_query_internal(self, query: str, task_type: str, user_answer: str) -> Dict:
        try:
            start_time = time.time()
            self.query_count += 1
            
            relevant_docs = self._retrieve_documents(query)
             # <<< 變更點 3：將任務相關資訊傳遞給 _build_prompt >>>
            prompt = self._build_prompt(query, relevant_docs, task_type, user_answer)
            response = self.model.generate_content(prompt)
            
            end_time = time.time()
            self.total_response_time += (end_time - start_time)

            # --- 解析模型的 JSON 回應 ---
            # 移除常見的 markdown 標籤 ```json ... ```
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            
            try:
                # 解析 JSON 字串為 Python 字典
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                error_message = f"無法解析模型的 JSON 回應。錯誤位置：{e.pos}，錯誤訊息：{e.msg}。原始回應：\n{response.text}"
                logger.error(error_message)  # 使用 logger.error 記錄錯誤
                return {"error": error_message, "raw_response": response.text}  # 返回更詳細的錯誤訊息

        except Exception as e:
            logger.error(f"處理查詢時發生嚴重錯誤: {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}

    # get_performance_metrics 方法保持不變
    def get_performance_metrics(self) -> Dict:
        """
        獲取性能指標
        
        Returns:
            Dict: 性能指標
        """
        # <<< 變更點：從 lru_cache 獲取快取資訊
        cache_info = self.process_query.cache_info()
        hits = cache_info.hits
        misses = cache_info.misses
        total_lookups = hits + misses

        return {
            "total_queries_processed": self.query_count, # 實際執行（未命中快取）的次數
            "total_queries_received": total_lookups, # 總共收到的查詢次數（包含命中快取）
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": hits / total_lookups if total_lookups > 0 else 0,
            "average_response_time_ms": (self.total_response_time / self.query_count * 1000) if self.query_count > 0 else 0,
            "total_documents_in_db": len(self.vector_db)
        }


# <<< 變更點：修改 main 函式以展示新功能
def main():
    """主函數"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("請設置 GOOGLE_API_KEY 環境變數")
    
    my_docs_directory = "my_textbook_data"
    if not os.path.exists(my_docs_directory):
        os.makedirs(my_docs_directory)
        logger.warning(f"已建立資料夾 '{my_docs_directory}'。請將你的 .md 或 .txt 檔案放入其中後再執行一次。")
        # 為了能執行，創建一個範例文件
        with open(os.path.join(my_docs_directory, "chapter_1_diagnostics.md"), "w", encoding="utf-8") as f:
            f.write("""Molecular diagnostics is defined as the use of the techniques of molecular biology for the purpose of prevention, diagnosis, follow-up, or prognosis of disease. The invention of the polymerase chain reaction (PCR) in 1985 was a key milestone. Pharmacogenetics is the study of variation in drug metabolism between individuals.""")
        # return

    rag_system = GeminiRAG(api_key, docs_dir=my_docs_directory, db_path="my_textbook.pkl")

     # --- 測試 1: 解析法律概念 ---
    print("\n--- 任務：解析法律概念 ---")
    concept_query = "何謂「行政自我拘束原則」？"
    print(f"問題：{concept_query}")
    user_answer = ""
    response1 = rag_system.process_query(query=concept_query, task_type="explain_concept", user_answer=user_answer)
    print("回答 (JSON):")
    print(json.dumps(response1, indent=2, ensure_ascii=False))

    # --- 測試 2: 批改申論題 ---
    print("\n--- 任務：批改申論題 ---")
    essay_question = "何謂「行政自我拘束原則」？其理論基礎及具體適用之要件為何？請申論之。"
    user_answer = "行政自我拘束原則就是說行政機關要遵守自己訂的規定，這跟平等原則有關。如果之前都這樣做，以後也要這樣做。"
    print(f"題目：{essay_question}")
    print(f"學生答案：{user_answer}")
    
    response2 = rag_system.process_query(
        query=essay_question,
        task_type="grade_essay",  # <<-- 使用正確的 task_type
        user_answer=user_answer
    )
    print("評語 (JSON):")
    print(json.dumps(response2, indent=2, ensure_ascii=False))


    # 輸出性能指標
    metrics = rag_system.get_performance_metrics()
    print("\n\n--- 性能指標 ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    print(f"Cache Info: {rag_system.process_query.cache_info()}")

if __name__ == "__main__":
    main()