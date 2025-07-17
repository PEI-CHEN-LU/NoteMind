# -*- coding: utf-8 -*-

from openai import AsyncOpenAI, OpenAI
import DBConfig

LLM_API_KEY = DBConfig.LLM_API_KEY
LLM_API_BASE = DBConfig.LLM_API_BASE
LLM_MODEL_NAME = DBConfig.LLM_MODEL_NAME

def ask_LLM(context, question_zh):
    SYSTEM_PROMPT = """你是一個專業的資訊分析員，負責判斷問題是否與提供的上下文有關。請僅回傳 'True' 或 'False'。"""

    # 使用者提示詞：只回傳 True 或 False
    USER_PROMPT = f"""
    請根據以下內容進行判斷：

    <context>
    {context}
    </context>

    <question>
    {question_zh}
    </question>

    請問：<question> 中的問題，是否可以根據 <context> 的內容回答？

    如果：
    - 問題與 context 內容有關、可找到答案 → 請回傳 True
    - 問題與 context 無關、找不到對應資訊 → 請回傳 False

    請注意：
    - 不需要說明理由或提供額外資訊。
    - 僅允許輸出 'True' 或 'False'。
    """
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
    
    if 'False' in response.choices[0].message.content:
        return '找不到相關資訊'
    
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
    print("response : \n", response.choices[0].message.content)
    return response.choices[0].message.content


