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

# 確保索引與文本同步
if index.ntotal != len(stored_texts):
    raise ValueError(f"索引的向量數量 ({index.ntotal}) 與文本數量 ({len(stored_texts)}) 不匹配！")

# 用戶輸入處理函數
def preprocess_and_encode(input_text):
    # 分割句子並向量化
    sentences = nltk.sent_tokenize(input_text)
    sentence_vectors = model.encode(sentences)
    return sentences, sentence_vectors

# 查詢向量比對函數
def query_similar_vectors(input_text, n_probes=2, top_k=3):
    sentences, vectors = preprocess_and_encode(input_text)
    index.nprobe = n_probes  # 設定檢索的區塊數

    results = []
    for i, vector in enumerate(vectors):
        D, I = index.search(np.array([vector], dtype=np.float32), top_k)
        for j in range(len(I[0])):
            closest_index = int(I[0][j])
            if closest_index < 0 or closest_index >= len(stored_texts):  # 檢查索引範圍
                print(f"警告：檢索到的索引 {closest_index} 超出範圍，已跳過！")
                continue
            results.append((sentences[i], stored_texts[closest_index], D[0][j]))
    return results

# 主程式執行
input_text = input("請輸入一句話：")
try:
    results = query_similar_vectors(input_text, n_probes=5, top_k=3)

    # 將結果寫入檔案
    with open("text.txt", "a", encoding="utf-8") as f:
        f.write("輸入文本: " + input_text + "\n")
        f.write("相似結果:\n")
        for sentence, closest_text, distance in results:
            f.write(f"句子: {sentence}\n")
            f.write(f"最相近文本: {closest_text}\n")
            f.write(f"距離: {distance:.4f}\n")
            f.write("-" * 50 + "\n")

    print("結果已寫入 text.txt！")
except ValueError as e:
    print(f"錯誤：{e}")
