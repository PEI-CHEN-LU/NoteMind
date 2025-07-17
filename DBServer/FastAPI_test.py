# -*- coding: utf-8 -*-

# import sys
# import os
# sys.path.append(os.path.dirname(__file__))
# print(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAI
from pymilvus import MilvusClient

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    from ImageProcess import change_image_path_to_base64
except:
    from DBServer import change_image_path_to_base64

try:
    import DBConfig
except:
    from DBServer import DBConfig

app = FastAPI()

# 配置常數
LLM_API_KEY = DBConfig.LLM_API_KEY
LLM_API_BASE = DBConfig.LLM_API_BASE
LLM_MODEL_NAME = DBConfig.LLM_MODEL_NAME

EMBEDDING_API_KEY = DBConfig.EMBEDDING_API_KEY
EMBEDDING_API_BASE = DBConfig.EMBEDDING_API_BASE
EMBEDDING_MODEL_NAME = DBConfig.EMBEDDING_MODEL_NAME

MILVUS_BASE = DBConfig.MILVUS_BASE
MILVUS_USER = DBConfig.MILVUS_USER
MILVUS_PASSWORD = DBConfig.MILVUS_PASSWORD

DB_NAME = DBConfig.DB_NAME
COLLECTION_NAME = DBConfig.COLLECTION_NAME
DIMENSION = DBConfig.DIMENSION
BATCH_SIZE = DBConfig.BATCH_SIZE

# 建立執行緒池來處理同步函數
executor = ThreadPoolExecutor()

# 建立 async client
openai_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
embedding_client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE)

# 儲存進行中的任務
active_tasks = {}

class ChatRequest(BaseModel):
    text: str

class TaskResponse(BaseModel):
    task_id: str

@app.post("/chat", response_model=TaskResponse)
async def chat(request: ChatRequest):
    task_id = f"task_{len(active_tasks)}"
    task = asyncio.create_task(user_chat_async(request.text, task_id))
    active_tasks[task_id] = task
    return {"task_id": task_id, "status": "running"}

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    task = active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.cancel()
    return {"status": "Task cancelled"}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task = active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.done():
        try:
            result = task.result()
            del active_tasks[task_id]
            return {"status": "completed", "response_msg": result}
        except asyncio.CancelledError:
            del active_tasks[task_id]
            return {"status": "cancelled"}
        except Exception as e:
            del active_tasks[task_id]
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "running"}

# 包裝同步的 post_embedding_model 為 async
def post_embedding_model_sync(text):
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE)
    responses = client.embeddings.create(input=text, model=EMBEDDING_MODEL_NAME)
    return [res_data.embedding for res_data in responses.data]

async def post_embedding_model_async(text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, post_embedding_model_sync, text)

# 包裝同步的 user_chat 為 async
def user_chat_sync(input_str, milvus_client, collection_name):
    question_zh = input_str
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[post_embedding_model_sync(question_zh)[0]],
        limit=50,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    print('search_res : \n', retrieved_lines_with_distances)

    context = "\n".join([line[0] for line in retrieved_lines_with_distances])
    # SYSTEM_PROMPT = """列出並整理所有符合的相似要點，並用中文回答我"""
    # USER_PROMPT = f"""
    # Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    # <context>
    # {context}
    # </context>
    # <question>
    # {question_zh}
    # </question>
    # """
    
        # 系統提示詞：定義角色與任務
    SYSTEM_PROMPT = """你是一個專業的資訊整理員，請根據提供的上下文列出並整理所有符合的相似要點，並以清晰易懂的中文回答。"""

    # 使用者提示詞：明確要求格式
    USER_PROMPT = f"""
    請根據以下<context>中的資訊，回答<question>中的問題。

    <context>
    {context}
    </context>

    <question>
    {question_zh}
    </question>

    請求：
    - 只根據<context>提供的內容回答(包含圖片路徑)。
    - 若無相關資訊，請回覆「找不到相關資訊」。
    - 請使用條列式或分段方式，讓回答清晰易讀。
    """
    openai_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    response = openai_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print('question :', question_zh)
    print("response : \n", response.choices[0].message.content)
    response_msg = question_zh + '\n' + response.choices[0].message.content
    response_msg = change_image_path_to_base64(response_msg)
    print(type(response_msg))
    return response_msg


async def user_chat_async(input_str: str, task_id: str):
    milvus_client = MilvusClient(uri=MILVUS_BASE, db_name=DB_NAME, user=MILVUS_USER, password=MILVUS_PASSWORD)

    try:
        # 用 executor 執行同步函數
        loop = asyncio.get_event_loop()
        response_msg = await loop.run_in_executor(executor, user_chat_sync, input_str, milvus_client, COLLECTION_NAME)
        return response_msg
    except asyncio.CancelledError:
        print(f"[Task {task_id}] 已被取消")
        return "Request was cancelled"
    except Exception as e:
        print(f"[Task {task_id}] 發生錯誤: {e}")
        return f"Error: {str(e)}"
    
import nest_asyncio
import asyncio
from uvicorn import Config, Server

# 解決 event loop 衝突問題
nest_asyncio.apply()

# 啟動 FastAPI 伺服器
async def run_server():
    config = Config(app=app, host="127.0.0.1", port=8000, loop="asyncio")
    server = Server(config)
    await server.serve()

# 執行伺服器
asyncio.run(run_server())


# python I:\2025\AutoCAMBot\AutoCam\AutoCAM2.0\main.py

#%%

def ask_LLM(USER_PROMPT):
    SYSTEM_PROMPT = """你是一位全能助理"""
    openai_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    response = openai_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        temperature=1.5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print("response : \n", response.choices[0].message.content)
    return response.choices[0].message.content