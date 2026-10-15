# Improved Authentic RAG (ข้อ 2 ของ Assignment 4)

ปรับปรุง/ขยายระบบจาก [`../baseline`](../baseline) ตามเงื่อนไขของโจทย์: แก้เฉพาะส่วน
**chunking, search algorithm, scoring, reranking** — **ไม่แตะ LLM ที่ generate คำตอบสุดท้าย**
(ยังใช้ `qwen/qwen-2.5-72b-instruct` ผ่าน OpenRouter เหมือนเดิม) และยังใช้ embedding model
เดิม (`BAAI/bge-m3`) เหมือนเดิมด้วย เพื่อให้มั่นใจว่าความแตกต่างของผลลัพธ์มาจากจุดที่ปรับปรุงจริงๆ

## 3 จุดที่ปรับปรุง

### 1. Chunking — จาก fixed-size เป็น structure-aware + overlap
| | Baseline | Improved |
|---|---|---|
| วิธีแบ่ง | `MarkdownNodeParser(chunk_size=256)` ตัดตามจำนวนตัวอักษรล้วนๆ | แบ่งตามหัวข้อ markdown (`#`,`##`,`###`) ก่อน แล้วค่อยแบ่งซ้ำด้วย `RecursiveCharacterTextSplitter` ถ้า section ยาวเกิน |
| overlap | ไม่มี | มี overlap 80 ตัวอักษร ระหว่าง chunk ที่ถูกแบ่งซ้ำ |
| ปัญหาที่แก้ | chunk อาจตัดกลางหัวข้อ เช่น ประโยคสุดท้ายของ "สาเหตุ" หลุดไปปนกับ "อาการ" | แต่ละ chunk อยู่ในหัวข้อเดียวกันครบถ้วน ไม่ตัดกลางประโยคสำคัญ |

### 2. Search algorithm — ดึง candidate pool กว้างขึ้นก่อนตัดสินใจ
Baseline ดึงผลค้นหาแค่ `k*2` (เช่น k=5 → 10 รายการ) แล้วให้ RRF ตัดสินอันดับสุดท้ายทันที
Improved ดึงมากขึ้นเป็น `candidate_pool_size=20` รายการ/วิธี (BM25 และ vector) แล้วรวมด้วย RRF
เพื่อให้ได้ "รายชื่อผู้เข้ารอบ" ที่กว้างและไม่พลาดเอกสารที่เกี่ยวข้อง ก่อนส่งต่อให้ reranker ตัดสินจริง

### 3. Scoring / Reranking — คะแนนจาก cross-encoder แทนคะแนน RRF
RRF ให้คะแนนจาก "อันดับ" (`1/(k+rank)`) เท่านั้น ไม่ได้ดูว่าเนื้อหากับคำถามเกี่ยวข้องกันจริงแค่ไหน
Improved เพิ่มขั้น **reranking** โดยใช้ cross-encoder (`BAAI/bge-reranker-v2-m3` รองรับภาษาไทย)
คำนวณคะแนนความเกี่ยวข้องระหว่าง query กับเนื้อหาแต่ละ candidate โดยตรง แล้วใช้คะแนนนี้เลือก
top-k สุดท้ายที่จะส่งให้ LLM — แม่นกว่าคะแนน RRF ที่เป็นแค่ตัวแทนของอันดับ

```
Query
  │
  ├─ sparse_search(k=20)  ─┐
  ├─ dense_search(k=20)   ─┼─ RRF fusion ─▶ candidate pool (20 รายการ)
  │                        │
  │                        ▼
  │              cross-encoder reranker (BAAI/bge-reranker-v2-m3)
  │                        │
  │                        ▼
  │              top-k ตามคะแนน rerank (ไม่ใช่คะแนน RRF)
  │                        │
  └────────────────────────┴──▶ Qwen 2.5 72B สรุปคำตอบ (เหมือน baseline ทุกประการ)
```

## การติดตั้งและรัน

ใช้ environment เดียวกับ baseline ได้เลย (ติดตั้ง dependency เพิ่มแค่ reranker model ซึ่งโหลดอัตโนมัติ
ผ่าน `sentence-transformers` ตอนรันครั้งแรก ไม่ต้อง pip install เพิ่ม)

```bash
cd improved
pip install -r requirements.txt   # เผื่อยังไม่มี environment จาก baseline

export OPENROUTER_API_KEY='your_api_key_here'

# 1. นำเข้าเอกสารด้วย chunking แบบใหม่ (เก็บลง index ชุด v2 แยกจาก baseline)
python improvedRAG.py

# 2. ค้นหา + rerank + ตอบคำถาม
python improvedSearchRAG.py
```

ผลลัพธ์จะถูกบันทึกไว้ที่ `improved_rag_search_results.json` เทียบกับ
`../baseline/authentic_rag_search_results.json` ได้โดยตรง (คำถามชุดเดียวกัน)

ดูการเปรียบเทียบผลลัพธ์แบบละเอียดได้ที่ [`../COMPARISON.md`](../COMPARISON.md)
