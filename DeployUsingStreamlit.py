'''import streamlit as st
from langchain_openai import ChatOpenAI
import sys
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

st.title('Ari`s First Retrieval Augmented Generation Web App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)

# Streamlit 应用程序界面
def main():
    st.title('Ari`s First Retrieval Augmented Generation Web App')
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        # 调用 respond 函数获取回答
        answer = generate_response(prompt, openai_api_key)
        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   



def get_vectordb():
    # 加载数据库
    vectordb = Chroma(
        persist_directory='vector_db/chroma',  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=OpenAIEmbeddings()
    )
    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str,openai_api_key:str):
    vectordb = get_vectordb()
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0.6,openai_api_key = openai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']


#不带历史记录的问答链
def get_qa_chain(question:str,openai_api_key:str):
    vectordb = get_vectordb()
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])'''



import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory

st.title('Ari`s Retrieval Augmented Generation Web App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# 获取向量数据库
def get_vectordb():
    vectordb = Chroma(
        persist_directory='vector_db/chroma',
        embedding_function=OpenAIEmbeddings()
    )
    return vectordb

# 生成回答
def generate_response(input_text, openai_api_key):
    llm = ChatOpenAI(temperature=0.6, openai_api_key=openai_api_key)
    response = llm.invoke(input_text)
    st.info(response)
    return response


# 带有历史记录的问答链
def get_chat_qa_chain(question, openai_api_key):
    vectordb = get_vectordb()
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6, openai_api_key=openai_api_key)

    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6, openai_api_key=openai_api_key),
        retriever=vectordb.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    result = qa({"question": question})
    return result['answer']

# 不带历史记录的问答链
def get_qa_chain(question, openai_api_key):
    vectordb = get_vectordb()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
    {context}
    问题: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb,
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]




# 选择对话模式
selected_method = st.radio(
    "你想选择哪种模式进行对话？",
    ["None", "qa_chain", "chat_qa_chain"],
    captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"]
)

if selected_method == "qa_chain":
    question = st.text_input("请输入你的问题:")
    if question:
        response = get_qa_chain(question, openai_api_key)
        st.write(response)

elif selected_method == "chat_qa_chain":
    question = st.text_input("请输入你的问题:")
    if question:
        response = get_chat_qa_chain(question, openai_api_key)
        st.write(response)





