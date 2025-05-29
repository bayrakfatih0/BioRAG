import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
st.title("Biology AI for students")

loader = PyPDFLoader("biology.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
documents = text_splitter.split_documents(data)
print("total", len(documents))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)

retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs = {"k" : 7})


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3,max_tokens=500)

query = st.chat_input("Biyoloji için buradayım..")

prompt = query

system_prompt = ("""Sen, 12. sınıf öğrencilerine biyoloji konusunda uzman desteği sağlayan bir yapay zeka asistanısın.
                    Öğrencilere, üniversiteye hazırlık sürecinde veya okul derslerine destek olacak şekilde konuları açık, anlaşılır ve sade bir dille anlatmalısın.
                    Aynı zamanda bilimsel doğruluk ve akademik ciddiyet taşımalısın.

                    Öğrenciler sana konu özeti, kavram açıklaması, örnek soru çözümü veya belirli biyolojik süreçlerin açıklamasını sorabilir.
                    Cevaplarını verirken:

                    Önce konuyu kısa ve net bir şekilde tanıt,

                    Ardından gerekli biyolojik terimleri açıkla,

                    Gerekirse örneklerle destekle (günlük yaşamdan veya sınav tipi örnek sorular),

                    Her zaman öğrencinin seviyesine uygun konuş (karmaşık terimleri sadeleştir),

                    Öğrenciyi öğrenmeye teşvik eden, motive edici bir dil kullan.

                    Eğer öğrenci net olmayan bir soru sorarsa, açıklama iste veya soruyu netleştirici sorular sor.
                    Gereksiz detaylardan kaçın, odak her zaman öğrencinin konuyu anlaması ve sınava yönelik doğru hazırlanması olmalı.
                 
                    {context} """)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" ,system_prompt),
        ("human","{input}")
    ]
)

if query:

    answer_question_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, answer_question_chain)
    response = rag_chain.invoke({"input": query})

    st.write(response["answer"])