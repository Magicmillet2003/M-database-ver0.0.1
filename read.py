import faiss
from sentence_transformers import SentenceTransformer
import pickle
import nltk
import numpy as np

# 初始化模型
nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')

# 載入索引和文本
index = faiss.read_index("vector_index.faiss")
with open("texts.pkl", "rb") as f:
    stored_texts = pickle.load(f)

# 設置距離閾值
DISTANCE_THRESHOLD = 1.0  # 距離閾值（越小越嚴格）

# 用戶輸入處理函數
def preprocess_and_encode(input_text):
    """
    預處理並將輸入文本轉換為向量
    """
    sentences = nltk.sent_tokenize(input_text)  # 句子分割
    sentence_vectors = model.encode(sentences)  # 向量化
    return sentences, sentence_vectors

# 查詢向量比對函數
def query_similar_vectors(input_text, n_probes=5, top_k=3):
    """
    查詢相似向量，並根據閾值判斷是否返回結果
    """
    sentences, vectors = preprocess_and_encode(input_text)
    index.nprobe = n_probes  # 設定檢索的區塊數

    results = []
    for i, vector in enumerate(vectors):
        D, I = index.search(np.array([vector], dtype=np.float32), top_k)
        # 判斷距離是否小於閾值
        if len(D[0]) > 0 and D[0][0] < DISTANCE_THRESHOLD:  # 最近的距離
            closest_index = int(I[0][0])
            results.append((sentences[i], stored_texts[closest_index], D[0][0]))
        else:
            results.append((sentences[i], "找不到相關資料", None))
    return results

# 主程式執行
input_text = input("請輸入一句話：")
results = query_similar_vectors(input_text, n_probes=5, top_k=3)

# 將結果寫入檔案
with open("text.txt", "a", encoding="utf-8") as f:
    f.write("輸入文本: " + input_text + "\n")
    f.write("相似結果:\n")
    for sentence, closest_text, distance in results:
        f.write(f"句子: {sentence}\n")
        if closest_text == "找不到相關資料":
            f.write("最相近文本: 找不到相關資料\n")
        else:
            f.write(f"最相近文本: {closest_text}\n")
            f.write(f"距離: {distance:.4f}\n")
        f.write("-" * 50 + "\n")

print("結果已寫入 text.txt！")
