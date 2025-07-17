LLM_API_KEY = "token-abc123"
LLM_API_BASE = "http://10.12.2.101:3332/v1"
LLM_MODEL_NAME = "gemma3-27B"

EMBEDDING_API_KEY  = "token-abc123"
EMBEDDING_API_BASE = "http://10.12.1.99:3251/v1"
EMBEDDING_MODEL_NAME = "bge-m3"

MILVUS_BASE = "http://10.12.1.99:19530"
MILVUS_USER = 'AutoCAM_admin'
MILVUS_PASSWORD = 'AutoCAM_admin'


DB_NAME = 'AutoCAM'
COLLECTION_NAME = "test"
DIMENSION = 1024
BATCH_SIZE = 256


FOLDER_DB_MAP = {
        "ST處理": "st_process",
        "內層": "inner",
        "前處理": "preprocess",
        "外層": "outer",
        "成型": "cnc",
        "排版": "type_setting",
        "文字": "silk",
        "板框": "panel",
        "鑽孔": "drill",
        "防焊": "mask"
    }