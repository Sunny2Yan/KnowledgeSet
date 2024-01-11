# -*- coding: utf-8 -*-

from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.embeddings import OpenAIEmbeddings


class Indexing:
    """使用 `RecordManager` 来跟踪写入vectorstore中的文档（对每个文档计算哈希值）。作用：
    1. 避免将重复的内容写入vectorstore;
    2. 避免重写未更改的内容;
    3. 避免在未更改的内容上重新计算嵌入
    """
    def __init__(self, openai_api_key):
        self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def create_index(self):
        collection_name = "test_index"
        vectorstore = ElasticsearchStore(
            es_url="http://localhost:9200", index_name="test_index",
            embedding=self.embedding)

        namespace = f"elasticsearch/{collection_name}"
        record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql")

        record_manager.create_schema()  # 使用record manager之前创建schema
        doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
        doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})

        # 清空 index (cleanup=incremental, )
        # index([], record_manager, vectorstore,
        #       cleanup="full", source_id_key="source")

        return index([doc1, doc2], record_manager, vectorstore,
                     cleanup=None, source_id_key="source", )


if __name__ == '__main__':
    api_key = ""
    index = Indexing(api_key)
    print(index.create_index())