# Baseline Authentic RAG (ข้อ 1 ของ Assignment 4)

ระบบ Hybrid RAG (BM25 + Vector search ด้วย Reciprocal Rank Fusion) ที่สร้างตามโค้ดและคำอธิบายในสไลด์
"PART4-2026-KDAI_05177141-Natural Language Processing" หน้า 22 และ 25 (Authentic RAG)

โค้ดต้นฉบับอ้างอิงจาก: https://github.com/aekanun2020/2025-authenticRAG

## สถาปัตยกรรมของระบบ

```
corpus_input/*.md  --(1) Ingestion: authenticRAG.py--
                                 |
                1. MarkdownNodeParser แบ่ง chunk (chunk_size=256)
                2. เรียก LLM (Qwen 2.5 72B ผ่าน OpenRouter) สร้าง "context" สั้นๆ อธิบายว่า
                   chunk นี้อยู่ตรงไหนของเอกสาร (Anthropic Contextual Retrieval)
                3. เอา chunk + context ไปสร้าง embedding ด้วย BAAI/bge-m3
                4. บันทึกลง OpenSearch 2 index:
                     - anthropic-vector-index  (dense vector, สำหรับ kNN similarity search)
                     - anthropic-bm25-index    (text + analyzer, สำหรับ BM25 lexical search)
                                 |
                                 v
Query --(2) Retrieval+Generation: onlysearchAuthenticRAG.py--
                1. sparse_search()  -> ค้น anthropic-bm25-index ด้วย multi_match (BM25)
                2. dense_search()   -> ค้น anthropic-vector-index ด้วย knn
                3. rrf_fusion()     -> รวมอันดับผลลัพธ์ทั้ง 2 ทางด้วย Reciprocal Rank Fusion
                4. generate_response() -> ส่ง top-k chunk ที่ได้ให้ Qwen 2.5 72B สรุปเป็นคำตอบ
```

## สิ่งที่ต้องมีก่อนรัน

- Python 3.10+
- Docker Desktop (รัน OpenSearch)
- API Key จาก OpenRouter (เรียกโมเดล `qwen/qwen-2.5-72b-instruct`) — สมัครที่ https://openrouter.ai/keys

## ขั้นตอนการติดตั้งและรัน

```bash
# 1. สร้าง environment แล้วติดตั้ง dependencies
conda create -n advrag python=3.10
conda activate advrag
cd baseline
pip install -r requirements.txt

# 2. รัน OpenSearch ด้วย Docker
docker network create opensearch-net
docker run -d \
  --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "network.host=0.0.0.0" \
  opensearchproject/opensearch:2.19.1

# 3. ตั้งค่า API key
export OPENROUTER_API_KEY='your_api_key_here'

# 4. นำเข้าเอกสาร (ingestion: chunking + contextualization + embedding + indexing)
python authenticRAG.py

# 5. ค้นหาและตอบคำถาม (retrieval + generation)
python onlysearchAuthenticRAG.py
```

ผลลัพธ์การค้นหา/ตอบคำถามจะถูกบันทึกไว้ที่ `authentic_rag_search_results.json`

## ข้อมูลตัวอย่าง (corpus_input)

เอกสารตัวอย่างเป็นบทความภาษาไทยเกี่ยวกับโรคหัด หัดเยอรมัน และโรคตา (ต้อกระจก ฯลฯ)
ใช้ทดสอบคำถามเช่น "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?"
