import os
from openai import OpenAI

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import TextLoader


class ImprovedContextualRAG:
    """
    ปรับปรุงจาก baseline/authenticRAG.py (AnthropicStyleContextualRAG) เฉพาะส่วน Chunking:

      เดิม : MarkdownNodeParser(chunk_size=256) ตัด chunk ตามจำนวนตัวอักษรล้วนๆ
             ไม่สนใจโครงสร้างหัวข้อ อาจตัดกลางหัวข้อ "สาเหตุ" "อาการ" "การรักษา" ได้

      ใหม่ : 1) แบ่งตามโครงสร้างหัวข้อ markdown (#, ##, ###) ก่อน เพื่อให้แต่ละ chunk
                อยู่ในหัวข้อเดียวกันครบถ้วน (ไม่ผสมหัวข้อ ไม่ตัดครึ่งหัวข้อ)
             2) ถ้า section ไหนยังยาวเกิน chunk_size ค่อยแบ่งซ้ำด้วย
                RecursiveCharacterTextSplitter แบบมี overlap กันข้อมูลตรงรอยต่อหาย

    embedding model (BAAI/bge-m3) และ LLM ที่ generate คำตอบสุดท้าย (qwen-2.5-72b-instruct)
    เหมือนเดิมกับ baseline ทุกประการ ตามเงื่อนไขของโจทย์ข้อ 2 (ห้ามเปลี่ยน LLM ที่ generate คำตอบ)
    """

    def __init__(self, opensearch_host="localhost", opensearch_port=9200,
                 chunk_size=400, chunk_overlap=80):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        class EmbeddingsWrapper:
            def __init__(self, embed_model):
                self.embed_model = embed_model

            def embed_query(self, text):
                return self.embed_model.encode(text).tolist()

        self.encoder = EmbeddingsWrapper(self.embed_model)

        self.opensearch_client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            use_ssl=False
        )

        # index ชุดใหม่ (v2) แยกจาก baseline เพื่อเทียบผลโดยไม่ทับข้อมูลเดิม
        self.vector_index_name = "anthropic-vector-index-v2"
        self.bm25_index_name = "anthropic-bm25-index-v2"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        self._create_or_update_indices()

    def _create_or_update_indices(self):
        if not self.opensearch_client.indices.exists(index=self.vector_index_name):
            vector_settings = {
                "settings": {
                    "index.knn": True,
                    "index.knn.space_type": "cosinesimil"
                },
                "mappings": {
                    "properties": {
                        "embedding": {"type": "knn_vector", "dimension": self.embedding_dim},
                        "doc_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "chunk_id": {"type": "keyword"}
                    }
                }
            }
            self.opensearch_client.indices.create(index=self.vector_index_name, body=vector_settings)
            print(f"สร้าง vector index {self.vector_index_name} สำเร็จ")

        if not self.opensearch_client.indices.exists(index=self.bm25_index_name):
            bm25_settings = {
                "settings": {
                    "analysis": {"analyzer": {"default": {"type": "standard"}}},
                    "similarity": {"default": {"type": "BM25"}},
                    "index.queries.cache.enabled": False
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "contextualized_content": {"type": "text", "analyzer": "standard"},
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "original_index": {"type": "integer", "index": False}
                    }
                }
            }
            self.opensearch_client.indices.create(index=self.bm25_index_name, body=bm25_settings)
            print(f"สร้าง BM25 index {self.bm25_index_name} สำเร็จ")

    def call_qwen_api(self, prompt, max_tokens=500, temperature=0.2):
        try:
            completion = self.client.chat.completions.create(
                model="qwen/qwen-2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Qwen API: {e}")
            return f"Qwen API error: {str(e)}"

    def load_and_chunk_documents(self, md_paths):
        """Chunking แบบ structure-aware + overlap (ดูคำอธิบายที่ docstring ของคลาส)"""
        print(f"Loading documents from {len(md_paths)} files...")
        pages = []
        for path in md_paths:
            loader = TextLoader(path)
            pages.extend(loader.load())

        lc_chunks = []
        for page in pages:
            header_sections = self.header_splitter.split_text(page.page_content)
            if not header_sections:
                header_sections = [LangchainDocument(page_content=page.page_content, metadata={})]

            for section in header_sections:
                section_text = section.page_content.strip()
                if not section_text:
                    continue

                header_path = " > ".join(v for v in section.metadata.values() if v)

                if len(section_text) <= self.chunk_size:
                    sub_chunks = [section_text]
                else:
                    sub_chunks = self.recursive_splitter.split_text(section_text)

                for sub in sub_chunks:
                    if not sub.strip():
                        continue
                    metadata = dict(page.metadata)
                    if header_path:
                        metadata["header_path"] = header_path
                    lc_chunks.append(LangchainDocument(page_content=sub.strip(), metadata=metadata))

        print(f"Split documents into {len(lc_chunks)} chunks (structure-aware + overlap={self.chunk_overlap})")
        return lc_chunks

    def get_context_prompt(self, document_content, chunk_content):
        return f"""<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
"""

    def generate_context(self, document_content, chunk_content):
        prompt = self.get_context_prompt(document_content, chunk_content)
        context = self.call_qwen_api(prompt, max_tokens=100, temperature=0.1)
        return context.strip()

    def add_documents_with_context(self, docs):
        print(f"\nAdding {len(docs)} documents to OpenSearch with contextual information...")
        vector_bulk_data = []
        bm25_bulk_data = []

        for i, doc in enumerate(docs):
            full_doc_content = doc.page_content
            context = self.generate_context(full_doc_content, doc.page_content)

            header_path = doc.metadata.get("header_path", "")
            if header_path:
                context = f"[{header_path}] {context}"
            print(f"Generated context for chunk {i+1}/{len(docs)}: {context[:60]}...")

            contextualized_content = f"{doc.page_content}\n\n{context}"
            embedding = self.encoder.embed_query(contextualized_content)

            vector_bulk_data.append({"index": {"_index": self.vector_index_name, "_id": f"doc_{i}"}})
            vector_bulk_data.append({
                "embedding": embedding,
                "doc_id": f"doc_{i}",
                "content": doc.page_content,
                "chunk_id": i
            })

            bm25_bulk_data.append({"index": {"_index": self.bm25_index_name, "_id": f"doc_{i}"}})
            bm25_bulk_data.append({
                "content": doc.page_content,
                "contextualized_content": context,
                "doc_id": f"doc_{i}",
                "chunk_id": i,
                "original_index": i
            })

        if vector_bulk_data:
            response = self.opensearch_client.bulk(body=vector_bulk_data)
            if response.get("errors", True):
                print("Some errors occurred during vector bulk upload.")
            else:
                print(f"Successfully added {len(docs)} documents to vector index")
            self.opensearch_client.indices.refresh(index=self.vector_index_name)

        if bm25_bulk_data:
            response = self.opensearch_client.bulk(body=bm25_bulk_data)
            if response.get("errors", True):
                print("Some errors occurred during BM25 bulk upload.")
            else:
                print(f"Successfully added {len(docs)} documents to BM25 index")
            self.opensearch_client.indices.refresh(index=self.bm25_index_name)

        return len(docs)


def download_corpus():
    os.makedirs('./corpus_input', exist_ok=True)


def main():
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Please set it with: export OPENROUTER_API_KEY='your_api_key_here'")
        return

    download_corpus()

    md_paths = [
        "./corpus_input/1.md",
        "./corpus_input/2.md",
        "./corpus_input/44.md",
        "./corpus_input/5555.md",
    ]

    rag = ImprovedContextualRAG(
        opensearch_host="localhost",
        opensearch_port=9200,
        chunk_size=400,
        chunk_overlap=80
    )

    docs = rag.load_and_chunk_documents(md_paths)
    print(f"Loaded {len(docs)} chunks")

    rag.add_documents_with_context(docs)

    print("\nIngestion (improved) เสร็จสมบูรณ์ — ใช้ improvedSearchRAG.py ค้นหาต่อได้เลย")


if __name__ == "__main__":
    main()
