from sentence_transformers import SentenceTransformer

# 選一個模型，例如英文用 bge-base-en
model = SentenceTransformer("BAAI/bge-base-en")

def get_embedding(sentence):
    # 輸入要嵌入的句子
    embedding = model.encode(sentence)
    return embedding