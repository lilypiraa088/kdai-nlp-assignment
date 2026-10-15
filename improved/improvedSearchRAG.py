import os
import json
from openai import OpenAI

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer, CrossEncoder


class ImprovedSearchRAG:
    """
    ปรับปรุงจาก baseline/onlysearchAuthenticRAG.py (AuthenticSearchRAG) ในส่วน
    Search algorithm + Scoring/Reranking:

      เดิม : ดึงผล BM25 และ vector search มาแค่ k*2 ตัว แล้วใช้ RRF (Reciprocal Rank Fusion)
             ตัดสินอันดับสุดท้ายทันที — RRF ใช้แค่ "อันดับ" ไม่ได้ดูว่า query กับเนื้อหา
             เกี่ยวข้องกันจริงแค่ไหน

      ใหม่ : 1) [Search algorithm] ดึง candidate pool กว้างขึ้น (ค่าเริ่มต้น 20 รายการ/วิธี)
                แล้วรวมด้วย RRF เพื่อให้ได้ "รายชื่อผู้เข้ารอบ" ที่ไม่พลาดเอกสารที่เกี่ยวข้อง
             2) [Reranking + Scoring] ใช้ cross-encoder reranker (BAAI/bge-reranker-v2-m3)
                ให้คะแนนความเกี่ยวข้องระหว่าง query กับเนื้อหาแต่ละ candidate โดยตรง
                (ไม่ใช่แค่อันดับ) แล้วใช้คะแนนนี้เป็นตัวตัดสิน top-k สุดท้ายที่จะส่งให้ LLM

    embedding model (BAAI/bge-m3) และ LLM ที่ generate คำตอบสุดท้าย (qwen-2.5-72b-instruct)
    เหมือนเดิมกับ baseline ทุกประการ ตามเงื่อนไขของโจทย์ข้อ 2
    """

    def __init__(self, opensearch_host="localhost", opensearch_port=9200,
                 candidate_pool_size=20, reranker_model="BAAI/bge-reranker-v2-m3"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.embed_model = SentenceTransformer('BAAI/bge-m3')

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

        # ใช้ index ชุดใหม่ (v2) ที่ improvedRAG.py สร้างไว้ (chunking แบบปรับปรุง)
        self.vector_index_name = "anthropic-vector-index-v2"
        self.bm25_index_name = "anthropic-bm25-index-v2"

        self.rrf_k = 60
        self.candidate_pool_size = candidate_pool_size

        print(f"Loading reranker model: {reranker_model} ...")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

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

    def sparse_search(self, query, k=10):
        response = self.opensearch_client.search(
            index=self.bm25_index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "contextualized_content"],
                        "type": "best_fields"
                    }
                },
                "size": k
            }
        )
        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def dense_search(self, query, k=10):
        query_embedding = self.encoder.embed_query(query)
        response = self.opensearch_client.search(
            index=self.vector_index_name,
            body={
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": k}}},
                "size": k
            }
        )
        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def rrf_fusion(self, rankings, k=60):
        fused_scores = {}
        doc_sources = {}
        for results in rankings:
            for rank, (doc_id, score, source) in enumerate(results):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_sources[doc_id] = source
                fused_scores[doc_id] += 1.0 / (k + rank)
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_sources[doc_id]) for doc_id, score in sorted_results]

    def retrieve_candidates(self, query):
        """[Search algorithm] ดึง candidate pool กว้างขึ้นจาก BM25 + vector ก่อน
        แล้วค่อยปล่อยให้ reranker เป็นคนตัดสินใจสุดท้าย แทนที่จะอาศัย RRF ตัดสินตรงๆ"""
        sparse_results = self.sparse_search(query, k=self.candidate_pool_size)
        dense_results = self.dense_search(query, k=self.candidate_pool_size)
        fused = self.rrf_fusion([sparse_results, dense_results], k=self.rrf_k)
        return fused[:self.candidate_pool_size]

    def rerank(self, query, candidates, top_k=5):
        """[Reranking + Scoring] ให้ cross-encoder คำนวณคะแนนความเกี่ยวข้องตรงๆ ระหว่าง
        query กับเนื้อหาแต่ละ candidate (แม่นกว่าคะแนน RRF ที่มาจากอันดับล้วนๆ)"""
        if not candidates:
            return []

        pairs = [(query, doc.get("content", "")) for _, _, doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        scored = [
            (candidates[i][0], float(rerank_scores[i]), candidates[i][2])
            for i in range(len(candidates))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def hybrid_search(self, query, k=5):
        candidates = self.retrieve_candidates(query)
        print(f"\nRetrieved {len(candidates)} candidates from hybrid (BM25+vector+RRF) search")
        reranked = self.rerank(query, candidates, top_k=k)
        print(f"Reranked down to top-{len(reranked)} with cross-encoder")
        return reranked

    def truncate_context(self, context, max_tokens=4000):
        words = context.split()
        estimated_tokens = len(words) * 1.3
        if estimated_tokens <= max_tokens:
            return context
        target_words = int(max_tokens / 1.3)
        truncated_context = " ".join(words[:target_words])
        return truncated_context + " ...(truncated)"

    def search_for_question(self, question, k=5):
        results = self.hybrid_search(question, k=k)

        if not results:
            return {"question": question, "answer": "ไม่พบข้อมูลที่เกี่ยวข้อง", "results": []}

        print(f"\nFound {len(results)} relevant documents for question: '{question}'")

        context = ""
        search_results = []
        for i, (doc_id, score, doc) in enumerate(results, 1):
            print(f"\nDOCUMENT {i}/{len(results)}")
            print(f"Rerank score: {score}")
            print(f"Doc ID: {doc_id}")

            result_info = {
                "doc_id": doc_id,
                "rerank_score": score,
                "content": doc.get("content", ""),
                "context": doc.get("contextualized_content", "")
            }
            search_results.append(result_info)

            content = doc.get("content", "")
            context_info = doc.get("contextualized_content", "")
            context += f"Content: {content}\nContext: {context_info}\n\n"

        truncated_context = self.truncate_context(context)
        answer = self.generate_response(question, truncated_context)
        return {"question": question, "answer": answer, "results": search_results}

    def generate_response(self, question, context="", max_tokens=1000):
        if not context:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Question: {question}"""
        else:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Context information:
{context}

Question: {question}"""
        return self.call_qwen_api(prompt, max_tokens=max_tokens, temperature=0.6)

    def search_multiple_questions(self, questions, k=5):
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n\n===== คำถามที่ {i}/{len(questions)} =====")
            print(f"คำถาม: {question}")
            try:
                result = self.search_for_question(question, k=k)
                results.append(result)
                print(f"\nคำตอบ: {result['answer']}")
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการค้นหาคำถาม '{question}': {e}")
                results.append({"question": question, "answer": f"เกิดข้อผิดพลาด: {str(e)}", "results": []})
        return results

    def export_results_to_json(self, results, output_file="search_results.json"):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nบันทึกผลลัพธ์ไปยัง {output_file} สำเร็จ")


def main():
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Please set it with: export OPENROUTER_API_KEY='your_api_key_here'")
        return

    rag = ImprovedSearchRAG(opensearch_host="localhost", opensearch_port=9200)

    questions = [
        'โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?',
        'อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน',
        'ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?',
        'ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?',
        'โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?'
    ]

    results = rag.search_multiple_questions(questions, k=5)
    rag.export_results_to_json(results, "improved_rag_search_results.json")


if __name__ == "__main__":
    main()
