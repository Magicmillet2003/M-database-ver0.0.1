import faiss
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

# 初始化模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 載入索引和文本
index = faiss.read_index("vector_index.faiss")
with open("texts.pkl", "rb") as f:
    stored_texts = pickle.load(f)

# 用戶輸入
input_text = input("請輸入一句話：")
input_vector = model.encode([input_text])

# 查找最相近的向量
D, I = index.search(input_vector, 1)  # 查找最近的1個

# 確保找到結果
if I[0][0] == -1:
    print("找不到相似的向量，請嘗試其他輸入。")
else:
    # 確保索引是 int64 類型
    closest_index = int(I[0][0])
    closest_vector = index.reconstruct(closest_index)
    closest_text = stored_texts[closest_index]

    # 將結果寫入檔案
    with open("text.txt", "a", encoding="utf-8") as f:
        f.write("輸入文本: " + input_text + "\n")
        f.write("輸入向量: " + np.array2string(input_vector[0]) + "\n")
        f.write("最相近向量: " + np.array2string(closest_vector) + "\n")
        f.write("最相近文本: " + closest_text + "\n")
        f.write("-" * 50 + "\n")

    print("結果已寫入 text.txt！")
