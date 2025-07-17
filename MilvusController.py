from pymilvus import MilvusClient, connections, Collection
import DBConfig
from pymilvus import DataType
import re
from openai import OpenAI


MILVUS_BASE = DBConfig.MILVUS_BASE
MILVUS_USER = DBConfig.MILVUS_USER
MILVUS_PASSWORD = DBConfig.MILVUS_PASSWORD

DB_NAME = DBConfig.DB_NAME

collection_name = "NoteBookLM"

DIMENSION = DBConfig.DIMENSION

EMBEDDING_API_KEY  = DBConfig.EMBEDDING_API_KEY
EMBEDDING_API_BASE = DBConfig.EMBEDDING_API_BASE
EMBEDDING_MODEL_NAME = DBConfig.EMBEDDING_MODEL_NAME

def parse_milvus_uri(uri: str):
    # 支援 http(s)://host:port 格式
    match = re.match(r'^https?://([^:]+):(\d+)$', uri.strip())
    if not match:
        raise ValueError(f"Invalid Milvus URI: {uri}")
    host, port = match.groups()
    return host, port

host, port = parse_milvus_uri(MILVUS_BASE)
connections.connect(host=host, port=port, user=MILVUS_USER, password=MILVUS_PASSWORD, db_name= DB_NAME)
collection = Collection(name=collection_name)

def main():
    
    milvus_client = MilvusClient(uri = MILVUS_BASE, db_name = DB_NAME, user = MILVUS_USER, password = MILVUS_PASSWORD)
    create_db_collection(milvus_client, collection_name)

    host, port = parse_milvus_uri(MILVUS_BASE)

    connections.connect(host=host, port=port, user=MILVUS_USER, password=MILVUS_PASSWORD, db_name= DB_NAME)

    collection = Collection(name=collection_name)
    
    example_data = [
    [101, 102],  # topic_id
    [1001, 1002],  # file_id
    [[0.1]*DIMENSION, [0.2]*DIMENSION],  # embedding vectors
    ["This is a test sentence.", "Another example."]  # text field
    ]

    example_user_id = 2

    insert_data_to_partition(collection, example_user_id, example_data)

    example_query_vector = [0.01] * DIMENSION

    target_topic_id = 101

    file_id_list = [1001]

    search_result = search_data_by_partition(collection, 
                                            example_user_id, 
                                            example_query_vector, 
                                            target_topic_id, 
                                            file_id_list)

    connections.disconnect()

def create_db_collection(milvus_client, collection_name):
    '''
    創建collection到milvus db，如果存在會刪掉重建
    '''
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="topic_id", datatype=DataType.INT64)
    schema.add_field(field_name="file_id", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=64000)


    milvus_client.create_collection(
        collection_name=collection_name, 
        schema=schema,
        metric_type="IP",
        consistency_level="Strong")

    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name="embedding", metric_type="IP", index_type="AUTOINDEX", params={}
    )

    milvus_client.create_index(collection_name=collection_name, index_params=index_params)
    milvus_client.load_collection(collection_name=collection_name, replica_number=1)




def insert_data_to_partition(collection, user_id, data):
    '''
    根據partition加入資料，如果沒有該partition則先創建一個
    '''

    partition_name = str(user_id)

    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name=partition_name)
        print(f"Partition {partition_name} 建立成功")
    else:
        print(f"Partition {partition_name} 已存在")

    collection.insert(data, partition_name= partition_name)

def search_data_by_partition(collection, user_id, query_vector, target_topic_id, file_id_list):
    '''
    從特定partition 搜尋資料且指定topic_id和file_id
    '''
    partition_name = str(user_id)
    results = collection.search(
    data=[query_vector],
    anns_field='embedding',
    param={'nprobe': 10},
    limit=5,
    expr = f"topic_id == {target_topic_id} && file_id in {file_id_list}",
    output_fields=['text'],
    partition_names=[partition_name]  # ← 指定 partition
    )
    return results


def sliding_window(text, window_size=100, overlap=30):
   """
   對單一字符串進行滑動窗口切割，並避開圖片路徑（D:...\.(png|jpe?g|gif)）被切斷。
   
   參數:
       text (str): 要切割的字串
       window_size (int): 每個窗口大小（預設100）
       overlap (int): 窗口間的重疊大小（預設30）

   回傳:
       list: 切割後的子字串列表
   """
   step = window_size - overlap
   result = []
   start = 0
   title = text.split("##")[0]

   # 找出所有 D 開頭、png/jpg/gif/jpeg 結尾的圖片路徑
   image_pattern = r'D:[^\"\s]*?\.(?:png|jpe?g|gif)'
   image_positions = [(m.start(), m.end()) for m in re.finditer(image_pattern, text, re.IGNORECASE)]

   def adjust_position(pos):
       """ 如果 pos 落在圖片路徑中，則返回圖片結尾位置作為新的 start """
       for img_start, img_end in image_positions:
           if img_start <= pos < img_end:
               return img_end  # 跳過整張圖片，從圖片結尾開始
       return pos

   def adjust_end(pos, img_end_limit):
       """ 如果 pos 落在圖片路徑中，則延伸到圖片結尾 """
       for img_start, img_end in image_positions:
           if pos >= img_start and pos <= img_end:
               return min(img_end, img_end_limit)  # 不超過最大文本長度
       return pos

   while start < len(text):
       # Step 1: 調整 start 點：不能落在圖片中
       adjusted_start = adjust_position(start)
       if adjusted_start > start:
           start = adjusted_start

       # Step 2: 確認 end 點
       end = start + window_size
       if end > len(text):
           end = len(text)

       # Step 3: 調整 end 點：如果 end 在圖片中，就延伸到圖片結尾
       end = adjust_position(end)

       # Step 4: 加入結果
       result.append(title + text[start:end])

       # Step 5: 更新 start 到下一段起點（保持 overlap）
       start = start + step

   return result

def delete_vector(user_id, topic_id, file_id):
    # user_id, topic_id, file_id = 1, 1, 1

    # Step 1: 準備 partition 名稱（假設你用 user_id 做 partition）
    partition_name = str(user_id)  # 或 f"user_{user_id}"，視你當初建立的 partition 名稱而定

    # Step 2: 準備 expr 過濾 topic_id 與 file_id
    expr = f"topic_id == {topic_id} && file_id == {file_id}"

    # Step 3: 確認 partition 存在（可選）
    if collection.has_partition(partition_name):
        # Step 4: 在指定 partition 中執行 delete
        collection.delete(expr=expr, partition_name=partition_name)
        print(f"[INFO] 已刪除 user_id={user_id}, topic_id={topic_id}, file_id={file_id} 的資料")
    else:
        print(f"[WARNING] Partition '{partition_name}' 不存在，無法刪除資料")

def post_embedding_model(text):
    client = OpenAI(api_key = EMBEDDING_API_KEY, base_url = EMBEDDING_API_BASE)
    responses = client.embeddings.create(input = text, model = EMBEDDING_MODEL_NAME)
    return [res_data.embedding for res_data in responses.data]


def upload_file_in_milvus(current_user_id, topic_id, file_id, file_path):
    '''
    file_path = r'I:\\2025\\ThemeCatalog\\uploads\\1\\20250709_102141_BGA_pad__Contact_Pad-OuterBGACompensate.md'
    
    '''
    # milvus_client = MilvusClient(uri = MILVUS_BASE, db_name = DB_NAME, user = MILVUS_USER, password = MILVUS_PASSWORD)
    # create_db_collection(milvus_client, collection_name)

    # host, port = parse_milvus_uri(MILVUS_BASE)

    # connections.connect(host=host, port=port, user=MILVUS_USER, password=MILVUS_PASSWORD, db_name= DB_NAME)

    # collection = Collection(name=collection_name)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        file_str = file.read()
        
    split_list = sliding_window(file_str, window_size=100, overlap=30)
    
    embeddings = post_embedding_model([item for item in split_list])
    
    '''
    example_data = [
                        [101, 102],  # topic_id
                        [1001, 1002],  # file_id
                        [[0.1]*DIMENSION, [0.2]*DIMENSION],  # embedding vectors
                        ["This is a test sentence.", "Another example."]  # text field
                    ]
    '''
    data = [
                [topic_id] * len(split_list),
                [file_id] * len(split_list),
                embeddings,
                split_list,
            ]
    
    insert_data_to_partition(collection, current_user_id, data)
    
    
def search_similar_embeddings(current_user_id, topic_id, file_id_list, question):
    '''
    current_user_id, topic_id, file_id_list = 1, 1, [2]
    question = 'bga補償邏輯'
    '''
    ref_result = []
    # host, port = parse_milvus_uri(MILVUS_BASE)
    # connections.connect(host=host, port=port, user=MILVUS_USER, password=MILVUS_PASSWORD, db_name= DB_NAME)
    # collection = Collection(name=collection_name)
    embeddings = post_embedding_model([item for item in [question]])
    ref_result = search_data_by_partition(collection, current_user_id, embeddings[0], topic_id, file_id_list)
    
    # for result in ref_result:
    #     for hit in result:
    #         text = hit.entity.get('text')
    #         file_id = hit.entity.get('file_id')
    #         print(f"ID: {hit.id}, Distance: {hit.distance}, Text: {text}, File ID: {file_id}")
    # THRESHOLD = 0.7
    # valid_results = []

    # for result in ref_result:
    #     for hit in result:
    #         print ({
    #                 'id': hit.id,
    #                 'distance': hit.distance,
    #                 'text': hit.entity.get('text'),
    #                 'file_id': hit.entity.get('file_id')
    #         })
    #         if hit.distance >= THRESHOLD:
                
    #             valid_results.append({
    #                 'id': hit.id,
    #                 'distance': hit.distance,
    #                 'text': hit.entity.get('text'),
    #                 'file_id': hit.entity.get('file_id')
    #         })
    return ref_result