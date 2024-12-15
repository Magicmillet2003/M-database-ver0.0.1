# M-database-ver0.0.3
---------------------------------------------------------------------------------
faiss
用途: 快速最近鄰檢索（向量索引和查詢）。
下載指令:
若為 CPU 版本：
pip install faiss-cpu
---------------------------------------------------------------------------------
若為 GPU 版本：
pip install faiss-gpu
---------------------------------------------------------------------------------
pip install nltk   分割句子及文本處理
pip install pdfplumber  pdf轉換
pip install sentence-transformers 文本轉化為語義向量
pip install numpy 處理向量以及數據等
---------------------------------------------------------------------------------
一次安裝
pip install nltk pdfplumber sentence-transformers faiss-cpu numpy
---------------------------------------------------------------------------------
transformer.py now transform txt files to creat vector database .
read.py can now change ur question into vector then
read the hole files that transformer.py created and compare its vector and ur question's vector.
