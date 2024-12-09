import faiss
from sentence_transformers import SentenceTransformer
import pickle
import nltk
from nltk.corpus import stopwords
import re
import string

# 初始化
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))  # 停用詞表（可擴展）
custom_noise_words = {"ah", "um", "oh", "hmm"}  # 自定義不需要的詞
model = SentenceTransformer('all-MiniLM-L6-v2')

# 讀取文本檔案
input_file = "AI文本.txt"  # 替換為您的文本檔案名稱
with open(input_file, "r", encoding="utf-8") as f:
    raw_texts = [line.strip() for line in f if line.strip()]  # 移除空行

# 清理和處理文本的函數
def preprocess_text(text):
    # 句子分割
    sentences = nltk.sent_tokenize(text)
    processed_sentences = []
    for sentence in sentences:
        # 小寫化
        sentence = sentence.lower()
        # 移除標點符號
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        # 移除不需要的詞
        words = sentence.split()
        filtered_words = [
            word for word in words if word not in stop_words and word not in custom_noise_words
        ]
        if filtered_words:
            processed_sentences.append(" ".join(filtered_words))
    return processed_sentences

# 處理所有文本
processed_texts = []
for text in raw_texts:
    processed_texts.extend(preprocess_text(text))

# 將文本轉換為向量
embeddings = model.encode(processed_texts)

# 建立 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 儲存索引和文本
faiss.write_index(index, "vector_index.faiss")
with open("processed_texts.pkl", "wb") as f:
    pickle.dump(processed_texts, f)

print(f"向量和文本已儲存完成！處理的文本數量: {len(processed_texts)}")
