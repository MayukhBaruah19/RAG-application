from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA

def create_qa_chain(llm, retriever, top_k=3):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Use the given context to answer the question."),
        HumanMessagePromptTemplate.from_template(
            "Question: {question}\n\nContext: {context}"
        )
    ])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt":prompt}
    )
    return qa_chain


