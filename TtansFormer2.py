import os
import faiss
import pickle
import pdfplumber
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

# 確保 NLTK 資料已下載
nltk.download('punkt')

# 初始化模型
model = SentenceTransformer('all-MiniLM-L6-v2')
index_path = "vector_index.faiss"
texts_path = "texts.pkl"
stored_texts = []

# 載入或初始化索引
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    with open(texts_path, "rb") as f:
        stored_texts = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 假設嵌入向量的維度是 384

# 預處理文本函數
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)  # 句子分割
    return sentences

# 處理 PDF 檔案
def process_pdf_to_txt(pdf_path, txt_output_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(text)

# 將文本轉換為向量並加入索引
def process_txt_to_vectors(txt_path, index, stored_texts, threshold=1e-6):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = preprocess_text(text)
    vectors = model.encode(sentences)

    for sentence, vector in zip(sentences, vectors):
        # 檢查向量是否已存在
        D, I = index.search(np.array([vector], dtype=np.float32), 1)
        if I[0][0] != -1 and D[0][0] < threshold:
            # 覆蓋現有文本
            closest_index = int(I[0][0])
            stored_texts[closest_index] = sentence
            print(f"覆蓋現有向量：{sentence}")
        else:
            # 新增向量與文本
            index.add(np.array([vector], dtype=np.float32))
            stored_texts.append(sentence)
            print(f"新增向量：{sentence}")

# 主程式
input_file = input("請輸入 PDF 或 TXT 檔案的路徑：")

if input_file.lower().endswith(".pdf"):
    txt_output_path = os.path.splitext(input_file)[0] + ".txt"
    process_pdf_to_txt(input_file, txt_output_path)
    process_txt_to_vectors(txt_output_path, index, stored_texts, threshold=1e-6)

    # 儲存索引和文本
    faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(stored_texts, f)
    print(f"索引和文本已成功使用 PDF 轉置成 TXT 並儲存！目前索引的向量數量：{index.ntotal}")

elif input_file.lower().endswith(".txt"):
    process_txt_to_vectors(input_file, index, stored_texts, threshold=1e-6)

    # 儲存索引和文本
    faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(stored_texts, f)
    print(f"索引和文本已成功使用 TXT 儲存！目前索引的向量數量：{index.ntotal}")

else:
    print("目前僅支援 PDF 或 TXT 檔案！")
