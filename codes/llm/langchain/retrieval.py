# -*- coding: utf-8 -*-
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS


class RetrievalChain:
    def __init__(self, api_key):
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
        self.llm = ChatOpenAI(openai_api_key=api_key, temperature=0, )
        self.db = None

    def vector_store(self, text_dir: str):
        pages = PyPDFLoader(text_dir).load()  # 按页加载
        document = ' '.join([page.page_content for page in pages])
        raw_documents = Document(page_content=document,
                                 metadata={'source': text_dir})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, )
        documents = text_splitter.split_documents([raw_documents])

        self.db = FAISS.from_documents(documents, self.embeddings_model)  # 存到向量数据库
        # cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        #     self.embeddings_model, store, namespace=self.embeddings_model.model
        # )
        # db = FAISS.from_documents(documents, cached_embedder)  #

    def _match(self, query: str):
        # docs = self.db.similarity_search(query)
        # print(docs[0].page_content)
        embedding_vector = self.embeddings_model.embed_query(query)
        docs = self.db.similarity_search_by_vector(embedding_vector)

        return docs

    def retrieval(self, query: str):
        """使用搜索向量存储的方法，如相似性搜索和MMR，查询矢量存储中的文本 (对上面match的包装).
        """
        retriever = self.db.as_retriever(search_type="mmr",  # default: vector similarity
                                         search_kwargs={"k": 1})  # top k
        docs = retriever.get_relevant_documents(query)

        return docs

    def other_retrieval(self, query):
        """
        1. multi_query_retrieval
        使用LLM从不同的角度为给定的query生成多个query，对于每个query，检索一组相关文档，
        并在所有查询中获取唯一联合，以获得更大的潜在相关文档集。

        2. Contextual compression
        文本较大时，查询最相关的信息可能被埋没在大量不相关文本的文档中.可以使用给定上下文的
        压缩再查询，以便只返回相关信息。

        ...
        """
        from langchain.retrievers.multi_query import MultiQueryRetriever
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=self.db.as_retriever(), llm=self.llm, )
        multi_query_docs = retriever_from_llm.get_relevant_documents(query=query)

        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
        retriever = self.db.as_retriever()
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        compressed_docs = compression_retriever.get_relevant_documents(query)

        return compressed_docs


if __name__ == '__main__':
    api_key = ""
    text_dir = "xxx.pdf"
    retrieval = RetrievalChain(api_key)
    retrieval.vector_store(text_dir)
    print(retrieval.retrieval("数据集是怎么收集的？"))
    print(retrieval.other_retrieval("数据集是怎么收集的？"))
