import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import re
import nltk
import numpy as np
import pdfplumber

# 初始化模型和工具
nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    # 僅基於句子分割（句點、感嘆號、問號）
    sentences = nltk.sent_tokenize(text)  # 使用 NLTK 分句
    # 選擇性去除無意義字詞（例如 '的', '了', 等等）
    stopwords = {"的", "了", "呢", "吧", "啊", "哦", "嘛", "喲"}
    processed_sentences = []
    for sentence in sentences:
        # 去除停用字
        filtered_sentence = ''.join(word for word in sentence if word not in stopwords)
        processed_sentences.append(filtered_sentence.strip())
    return processed_sentences
def process_pdf_to_txt(pdf_path, txt_output_path):
    """
    將 PDF 檔案轉換為 TXT 檔案
    """
    with pdfplumber.open(pdf_path) as pdf, open(txt_output_path, "w", encoding="utf-8") as txt_file:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                txt_file.write(text + "\n")
    print(f"PDF 已轉換為 TXT，檔案儲存於：{txt_output_path}")

def process_txt_to_vectors(txt_path, index, texts):
    """
    將 TXT 檔案轉換為向量並新增到索引
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 預處理文本
    processed_texts = preprocess_text(content)
    embeddings = model.encode(processed_texts)
    
    # 更新索引和文本列表
    index.add(embeddings)
    texts.extend(processed_texts)
    print(f"TXT 檔案已轉換為向量並新增至索引，處理句子數：{len(processed_texts)}")

# 建立或載入索引
index_path = "vector_index.faiss"
texts_path = "texts.pkl"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    with open(texts_path, "rb") as f:
        stored_texts = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384 是 all-MiniLM-L6-v2 的嵌入向量維度
    stored_texts = []

# 主程式
input_file = input("請輸入 PDF 或 TXT 檔案的路徑：")

if input_file.lower().endswith(".pdf"):
    txt_output_path = os.path.splitext(input_file)[0] + ".txt"
    process_pdf_to_txt(input_file, txt_output_path)
    process_txt_to_vectors(txt_output_path, index, stored_texts,threshold=1e-6)
    
    # 儲存索引和文本
    faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(stored_texts, f)
    print(f"索引和文本已成功使用pdf轉置成txt並儲存！目前索引的向量數量：{index.ntotal}")
    
elif input_file.lower().endswith(".txt"):
    process_txt_to_vectors(input_file, index, stored_texts,threshold=1e-6)
    
    # 儲存索引和文本
    faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(stored_texts, f)
    print(f"索引和文本已成功使用txt儲存！目前索引的向量數量：{index.ntotal}")
    
else:
    print("目前僅支援 PDF 或 TXT 檔案！")



# faiss.write_index(index, index_path)
# with open(texts_path, "wb") as f:
#     pickle.dump(stored_texts, f)
# print(f"索引和文本已成功儲存！目前索引的向量數量：{index.ntotal}")