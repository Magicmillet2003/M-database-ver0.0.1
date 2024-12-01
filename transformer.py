import faiss
from sentence_transformers import SentenceTransformer
import pickle

# 初始化模型
model = SentenceTransformer('all-MiniLM-L6-v2')
#使用 sentence-transformers 提供的預訓練模型，將文本轉換為固定維度的向量
    
# 從文本檔案中讀取內容
input_file = "AI文本.txt"  # 替換為您的文本檔案名稱
with open(input_file, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]  # 移除空行
#逐行讀取文本檔案。清除每行前後的空白，並忽略空行。

# 將文本轉換為向量
embeddings = model.encode(texts)
#使用模型將文本轉換為嵌入向量（即高維向量表示）。這些向量捕捉了文本的語義信息，可以用於比較相似度

# 建立 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
#使用 FAISS 創建一個基於 L2 距離（歐氏距離）的索引。將生成的向量加入到索引中。

# 儲存索引和文本
faiss.write_index(index, "vector_index.faiss")
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)
#faiss.write_index 將索引存儲到檔案中，用於後續查詢。
#使用 pickle 將文本列表存儲到檔案，保持向量與原始文本的對應關係。

print(f"向量和文本已儲存完成！處理的文本數量: {len(texts)}")
