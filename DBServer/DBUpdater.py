# -*- coding: utf-8 -*-
#%%-----------------------------------------------------------------------------
import os
import re

from openai import OpenAI
from pymilvus import MilvusClient
from pymilvus import DataType
from tqdm import tqdm
import DBConfig


LLM_API_KEY = DBConfig.LLM_API_KEY
LLM_API_BASE = DBConfig.LLM_API_BASE
LLM_MODEL_NAME = DBConfig.LLM_MODEL_NAME

EMBEDDING_API_KEY  = DBConfig.EMBEDDING_API_KEY
EMBEDDING_API_BASE = DBConfig.EMBEDDING_API_BASE
EMBEDDING_MODEL_NAME = DBConfig.EMBEDDING_MODEL_NAME

MILVUS_BASE = DBConfig.MILVUS_BASE
MILVUS_USER = DBConfig.MILVUS_USER
MILVUS_PASSWORD = DBConfig.MILVUS_PASSWORD

# tiktoken_cache_dir = "./tiktoken_cache"
# cache_key = '9b5ad71b2ce5302211f9c61530b329a4922fc6a4'
# os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

#%%-----------------------------------------------------------------------------
def post_embedding_model(text):
    client = OpenAI(api_key = EMBEDDING_API_KEY, base_url = EMBEDDING_API_BASE)
    responses = client.embeddings.create(input = text, model = EMBEDDING_MODEL_NAME)
    return [res_data.embedding for res_data in responses.data]


#%%
def create_db_collection():
    DB_NAME = DBConfig.DB_NAME
    COLLECTION_NAME = DBConfig.COLLECTION_NAME
    DIMENSION = DBConfig.DIMENSION
    BATCH_SIZE = DBConfig.BATCH_SIZE

    milvus_client = MilvusClient(uri = MILVUS_BASE, db_name = DB_NAME, user = MILVUS_USER, password = MILVUS_PASSWORD)

    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=64000)
    # schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=64000)
    # schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=64000)
    # schema.add_field(field_name="act", datatype=DataType.VARCHAR, max_length=3200)
    # schema.add_field(field_name="class", datatype=DataType.VARCHAR, max_length=3200)


    milvus_client.create_collection(
        collection_name=COLLECTION_NAME, 
        schema=schema,
        metric_type="IP",
        consistency_level="Strong")

    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name="embedding", metric_type="IP", index_type="AUTOINDEX", params={}
    )

    # index_params.add_index(
    #     field_name="embedding",
    #     metric_type="IP",
    #     index_type="GPU_CAGRA",
    #     params={
    #         "intermediate_graph_degree": 32,
    #         "graph_degree": 64,
    #         "search_length": 1024
    #     }
    # )

    milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
    milvus_client.load_collection(collection_name=COLLECTION_NAME, replica_number=1)
#%%
def batch_insert_embeddings(text_list):
    
    DB_NAME = DBConfig.DB_NAME
    COLLECTION_NAME = DBConfig.COLLECTION_NAME
    DIMENSION = DBConfig.DIMENSION
    BATCH_SIZE = DBConfig.BATCH_SIZE
    milvus_client = MilvusClient(uri = MILVUS_BASE, db_name = DB_NAME, user = MILVUS_USER, password = MILVUS_PASSWORD)
    
    batch = []

    # Embed and insert in batches
    for i in tqdm(range(0, len(text_list))):
        batch.append(
            {
                "text": text_list[i],

            }
        )

        if len(batch) % BATCH_SIZE == 0 or i == len(text_list) - 1:
            embeddings = post_embedding_model([item["text"] for item in batch])

            for item, emb in zip(batch, embeddings):
                item["embedding"] = emb

            milvus_client.insert(collection_name=COLLECTION_NAME, data=batch)
            batch = []
#%%
def update_documentation(folder_path):
    # txt_path = './sorting_data/first_data'
    # table_path = './sorting_data/tabel_data'
    # folder_path = r'I:\2025\Anti系列\說明文件'
    text_list = []
  
    # for folder_path in folder_list:
    for root, dirs, files in os.walk(folder_path):
        file_topic = root.split('\\')[-1]
        for filename in files:
            file_path = os.path.join(root, filename) 

            if filename.endswith('.md'):
                # raw_name = filename[:-4].split(' ')[-1]
                raw_name = file_topic + '---' + filename.split('.')[0]
                text_list = read_file_md_foramt(file_path, text_list, raw_name)
            # text_list = read_file_line(file_path, text_list, raw_name)

    batch_insert_embeddings(text_list)

#%%
def update_code_info(directory_path):
    file_path_list = get_folder_files(directory_path)
    
    text_list = []
    # for folder_path in folder_list:
    for file_path in file_path_list:
        # raw_name = filename[:-4].split(' ')[-1]
        raw_name = file_path.split('\\')[4]
        text_list = read_code_foramt(file_path, text_list, raw_name)
        # text_list = read_file_line(file_path, text_list, raw_name)

    batch_insert_embeddings(text_list)

def get_folder_files(directory_path):
    # 遍歷目錄及其子目錄
    file_path_list = []
    error_file_path = []
    for root, dirs, files in os.walk(directory_path):
        if root.endswith('__pycache__') or '.venv' in root:
            continue
        for file in files:
            if not file.endswith('.py') or file == '__init__.py':
                continue
            file_path = os.path.join(root, file)  # 獲取檔案的完整路徑
            try:
                with open(file_path, 'r', encoding='utf-8') as f:  # 使用適當的編碼打開檔案
                    content = f.read()  # 讀取檔案內容
                    # print(f'檔案: {file_path}')  # 輸出檔案路徑和內容
                file_path_list.append(file_path)
            except Exception as e:
                error_file_path.append(file_path)
                # print(f'無法讀取檔案 {file_path}，錯誤: {e}')
    return file_path_list


def split_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # 使用正則表達式找到所有函數定義
    functions = re.split(r'\n(?=def )', code)
    return functions

#%%
def read_code_foramt(file_path, text_list, raw_name):
    functions = split_code(file_path)
    for function in functions:
        current_section = raw_name + '---' + function + '\n'
        text_list.append(current_section)
    return text_list
        

def read_file_md_foramt(file_path, text_list, raw_name):
    '''
    以##來做區分
    '''
    current_section = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()

            # 如果遇到以 ## 開頭的行
            if stripped_line.startswith('##'):
                # 如果之前已經有一個 section，先存起來
                if current_section is not None:
                    text_list.append(current_section)

                # 開始新的 section
                current_section = raw_name + '---' + stripped_line + '\n'
            elif current_section is not None:
                # 如果目前在某個 section 中，就把這一行加進去
                current_section += line  # 保留原始換行

        # 處理最後一個 section
        if current_section is not None:
            text_list.append(current_section)
    return text_list

def read_file_line(file_path, text_list, raw_name):
    with open(file_path, 'r', encoding='utf-8') as file:
                    # content = file.read()
                    # print(f'檔案名稱: {filename}')
                    # print('內容:')
                    # print(content)
                    # print('---------------------')
        for line in file:
                        # print(line.strip())  # 逐行讀取並去除換行符
            if len(line.strip()):
                text_list.append(raw_name+'---'+line.strip())
    return text_list

def rename_file(folder_path):
    # 確認資料夾存在
    if not os.path.isdir(folder_path):
        print(f"錯誤：路徑 '{folder_path}' 不存在或不是資料夾。")
        return

    # 遍歷資料夾中的檔案
    for filename in os.listdir(folder_path):
        # 取得副檔名
        name, ext = os.path.splitext(filename)
        
        # 如果是 .md 檔
        if ext.lower() == '.md':
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, name + '.txt')
            
            # 更名檔案
            os.rename(old_path, new_path)
            print(f"已更名：{filename} → {name}.txt")
            

def read_file_and_split(file_path):
    
    text_list = []
    current_section = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()

            # 如果遇到以 ## 開頭的行
            if stripped_line.startswith('##'):
                # 如果之前已經有一個 section，先存起來
                if current_section is not None:
                    text_list.append(current_section)
                    
                # 開始新的 section
                current_section = stripped_line + '\n'
            elif current_section is not None:
                # 如果目前在某個 section 中，就把這一行加進去
                current_section += line  # 保留原始換行

        # 處理最後一個 section
        if current_section is not None:
            text_list.append(current_section)
    return text_list


def insert_embeddings(current_user_id, topic_id, file_id, text_list):
    """
    將文本轉為 embedding 並插入 Milvus，根據 user_id 分配 partition
    
    Args:
        current_user_id (int): 使用者 ID
        topic_id (int): 主題 ID
        file_id (int): 文件 ID
        text_list (List[str]): 要嵌入的文本列表
    """

    # 從設定檔取得參數
    DB_NAME = DBConfig.DB_NAME
    COLLECTION_NAME = DBConfig.COLLECTION_NAME
    DIMENSION = DBConfig.DIMENSION
    BATCH_SIZE = DBConfig.BATCH_SIZE

    # Milvus 連線資訊
    milvus_client = MilvusClient(
        uri=DBConfig.MILVUS_BASE,
        db_name=DB_NAME,
        user=DBConfig.MILVUS_USER,
        password=DBConfig.MILVUS_PASSWORD
    )

    # 確保 partition 存在（以 user_id 為名）
    partition_tag = f"user_{current_user_id}"
    if not milvus_client.has_partition(collection_name=COLLECTION_NAME, partition_tag=partition_tag):
        milvus_client.create_partition(collection_name=COLLECTION_NAME, partition_tag=partition_tag)

    batch = []

    try:
        for i in tqdm(range(len(text_list)), desc="處理 Embedding"):
            batch.append({
                "text": text_list[i],
                "user_id": current_user_id,
                "topic_id": topic_id,
                "file_id": file_id,
            })

            # 到達批次大小或最後一筆時插入
            if len(batch) >= BATCH_SIZE or i == len(text_list) - 1:
                embeddings = post_embedding_model([item["text"] for item in batch])

                # 加入 embedding 欄位
                for item, emb in zip(batch, embeddings):
                    item["embedding"] = emb

                # 插入 Milvus 的指定 partition
                milvus_client.insert(collection_name=COLLECTION_NAME, data=batch, partition_tag=partition_tag)
                batch = []

    except Exception as e:
        print(f"[ERROR] 插入失敗: {e}")
        raise

def upload_file_in_milvus(current_user_id, topic_id, file_id, file_path):

    text_list = read_file_and_split(file_path)
    insert_embeddings(current_user_id, topic_id, file_id)

#%%

def main():
   
    
    create_db_collection()
    folder_path = r'I:\2025\AutoCAMBot\文檔'
    rename_file(folder_path)
    update_documentation(folder_path)
    
    directory_path = r'I:\2025\AutoCAMBot\程式'
    update_code_info(directory_path)