import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# 加載BERT模型和tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 文本轉向量的函式
def text_to_vector(text):
    """
    將輸入文本轉換為固定長度的向量。
    使用BERT模型提取CLS token的嵌入表示。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze()  # 取CLS向量
    return vector.numpy()

# 建立FAISS索引的函式
def create_index(vectors):
    """
    根據輸入的向量資料，建立FAISS索引。
    預設使用L2距離（歐幾里得距離）進行相似度檢索。
    """
    d = vectors.shape[1]  # 向量的維度
    index = faiss.IndexFlatL2(d)
    index.add(vectors)  # 添加向量到索引
    return index

# 主程式邏輯
if __name__ == "__main__":
    # 假設我們有一組文本資料
    texts = [
        "Hello, how are you?",
        "FAISS is a library for efficient similarity search.",
        "I love learning about machine learning."
    ]
    
    # 將文本轉換為向量
    vectors = np.array([text_to_vector(text) for text in texts])

    # 建立FAISS索引
    index = create_index(vectors)
    print(f"FAISS index contains {index.ntotal} vectors.")
    
    # 測試查詢
    query = "I want to learn about AI and machine learning."
    query_vector = text_to_vector(query)
    k = 1  # 找最近的3個向量
    D, I = index.search(np.array([query_vector]), k)
    
    # 輸出查詢結果
    print(f"Query: {query}")
    print("Most similar texts:")
    for idx, dist in zip(I[0], D[0]):
        print(f"Text: {texts[idx]}, Distance: {dist}")
    
    # 保存索引
    faiss.write_index(index, "vector_index.faiss")
    print("FAISS index saved to 'vector_index.faiss'")
