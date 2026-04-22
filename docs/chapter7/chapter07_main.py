import os

from dotenv import load_dotenv

from docs.chapter7.chapter07_my_llm import MySimpleAgent

# 加载环境变量
load_dotenv()


if __name__ == '__main__':

    # 正确读取方式
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    MODEL_ID = os.getenv("MODEL_ID")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


    # 实例化我们重写的客户端，并指定provider
    llm = MySimpleAgent(provider="modelscope",api_key=API_KEY, base_url=BASE_URL,model=MODEL_ID)

    message=[{"role": "user", "content": "你好，请介绍一下你自己。"}]
    response=llm.think(message)

    # 打印响应
    print("ModelScope Response:")
    for chunk in response:
        # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
        print(chunk, end="", flush=True)
        pass