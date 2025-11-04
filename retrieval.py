import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

current_dir=os.getcwd()
db_path=os.path.join(current_dir,"db")
persistent_directory=os.path.join(db_path,"chroma_db")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store=Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_directory
)
print("Vector store loaded successfully.")
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})

print("Retriever created successfully.")


llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

context          = "\n".join([doc.page_content for doc in retrieved_docs])
final_prompt     = prompt.format(context=context, question=question)

response        = llm.invoke(final_prompt)

print("Response from LLM:")
print(response.content)

# input by user and reply from the LLM based on the retrieved context with continuation 

while True:
    user_question = input("Enter your question (or 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break

    retrieved_docs = retriever.invoke(user_question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    final_prompt = prompt.format(context=context, question=user_question)

    response = llm.invoke(final_prompt)

    print("Response from LLM:")
    print(response.content)

