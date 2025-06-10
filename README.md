# RAG + LangChain 실습 레포 (with ChromaDB, Codespaces)

## 실습 모듈

1. **RAG/LangChain 이론과 환경 설정**  
   → `notebook/0_rag_intro_env.ipynb`

2. **문서 로딩/전처리**  
   → `notebook/1_data_ingestion.ipynb`

3. **임베딩 및 ChromaDB 벡터 스토어 구축**  
   → `notebook/2_embedding_chromadb.ipynb`

4. **검색 & 생성 (RAG 체인)**  
   → `notebook/3_retrieval_generation.ipynb`

5. **(심화) 평가 및 최적화**  
   → `notebook/4_advanced_evaluation.ipynb`

---

## 실습 환경

- GitHub Codespaces 권장
- Jupyter Notebook 기반 단계별 실습
- Streamlit 챗봇 예제 포함

---

## 시작하기

1. Codespace에서 requirements.txt 설치  
   ```bash
   pip install -r requirements.txt
   ```
2. `.env.example`을 복사해 `.env` 생성 후 OpenAI API 키 입력  
3. Jupyter로 각 모듈 노트북 실행  
4. (옵션) Streamlit 챗봇 실행  
   ```
   streamlit run app/streamlit_app.py
   ```

---

## 기타

- 샘플 문서는 `data/sample_docs.txt`
- ChromaDB는 로컬에 `./chroma` 폴더로 저장됨
- `.env` 파일은 절대 커밋 금지!