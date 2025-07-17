# -*- coding: utf-8 -*-
#%%-----------------------------------------------------------------------------
import os

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


DB_NAME = DBConfig.DB_NAME
COLLECTION_NAME = DBConfig.COLLECTION_NAME
DIMENSION = DBConfig.DIMENSION
BATCH_SIZE = DBConfig.BATCH_SIZE

# tiktoken_cache_dir = "./tiktoken_cache"
# cache_key = '9b5ad71b2ce5302211f9c61530b329a4922fc6a4'
# os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

#%%-----------------------------------------------------------------------------
def post_embedding_model(text):
    client = OpenAI(api_key = EMBEDDING_API_KEY, base_url = EMBEDDING_API_BASE)
    responses = client.embeddings.create(input = text, model = EMBEDDING_MODEL_NAME)
    return [res_data.embedding for res_data in responses.data]

    # %%
def user_chat(input_str, milvus_client, COLLECTION_NAME):
    question_zh = input_str
    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[
            post_embedding_model(question_zh)[0],
        ],
        limit=50, 
        search_params={"metric_type": "IP", "params": {}}, 
        output_fields=["text"],
    )

    # DQI_x_T_L = 0.6127370449920144
    # DQI_x_T_L = 0.0

    # search_res_DQI = []
    # for i, res_dict in enumerate(search_res[0]):
    #     if res_dict["distance"] > DQI_x_T_L:
    #         search_res_DQI.append(search_res[0][i])
    #     else:
    #         break
    
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    # retrieved_lines_with_distances = [
    #     (res["entity"]["text"], res["distance"]) for res in search_res_DQI
    # ]
    print('search_res : \n', retrieved_lines_with_distances)

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )
    SYSTEM_PROMPT = """列出並整理所有符合的相似要點，並用中文回答我"""
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question_zh}
    </question>
    """
    openai_client = OpenAI(api_key = LLM_API_KEY, base_url = LLM_API_BASE)
    response = openai_client.chat.completions.create(
        model = LLM_MODEL_NAME,
        temperature = 0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print('question :', question_zh)
    print("response : \n", response.choices[0].message.content)
    response_msg = question_zh + '\n' + response.choices[0].message.content
    return response_msg

#%%
def test():
    input_str = '寫出anti程式的流程圖'
    milvus_client = MilvusClient(uri = MILVUS_BASE, db_name = DB_NAME, user = MILVUS_USER, password = MILVUS_PASSWORD)
    
    user_chat(input_str, milvus_client, COLLECTION_NAME)
    
#%%
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive_data():
    data = request.get_json()
    # Process the data here
    print(data['text'])  # For debugging purposes, prints the received data

    milvus_client = MilvusClient(uri = MILVUS_BASE, db_name = DB_NAME, user = MILVUS_USER, password = MILVUS_PASSWORD)
    
    response_msg = user_chat(data['text'], milvus_client, COLLECTION_NAME)
    
    return jsonify({'status': 'success', 'response_msg': response_msg}), 200

if __name__ == '__main__':
    app.run(debug=False)
    
# python I:\2025\AutoCAMBot\AutoCam\AutoCAM2.0\main.py