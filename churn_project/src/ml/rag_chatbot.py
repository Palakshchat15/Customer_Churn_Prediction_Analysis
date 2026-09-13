import os
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class ChurnRAGChatbot:
    def __init__(self):
        """
        Initializes the RAG Chatbot using HuggingFace models.
        Requires HUGGINGFACEHUB_API_TOKEN in environment variables.
        """
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.report_path = self.project_root / "report" / "Data_Analysis_Report.md"
        
        print("Initializing Embeddings Model...")
        # Using a fast, free embedding model from HF
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.setup_rag()

    def setup_rag(self):
        report_dir = self.project_root / "report"
        md_files = list(report_dir.glob("*.md"))
        
        if not md_files:
            print(f"Warning: No markdown reports found in {report_dir}")
            return
            
        print(f"Loading {len(md_files)} report(s): {[f.name for f in md_files]}")
        
        from langchain_core.documents import Document
        docs = []
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": md_file.name}))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"  → {len(splits)} chunks indexed into FAISS vector store")
        
        print("Creating FAISS Vector Store...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

    def query(self, question: str) -> str:
        groq_token = os.getenv("GROQ_API_KEY")
        if not groq_token:
            return "Please set GROQ_API_KEY in your environment or .env file."
            
        from langchain_groq import ChatGroq

        # Using Groq's free, ultra-fast LLaMA3 inference API
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            api_key=groq_token
        )
        
        retriever = self.vector_store.as_retriever()
        
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context. 
            If you don't know the answer, just say that you don't know.
            
            Context:
            {context}
            
            Question: {input}
            
            Answer:"""
        )
        
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        retrieval_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        try:
            print("Querying LLM...")
            response = retrieval_chain.invoke(question)
            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during generation: {repr(e)}"

if __name__ == "__main__":
    bot = ChurnRAGChatbot()
    print("\nChatbot Ready! (Type 'quit' to exit)")
    while True:
        q = input("\nYou: ")
        if q.lower() == 'quit':
            break
        ans = bot.query(q)
        print(f"\nBot: {ans}")
