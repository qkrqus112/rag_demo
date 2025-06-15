# RAG + LangChain 실습 예제 (ChromaDB & Codespaces)

## 개요

이 저장소는 Retrieval-Augmented Generation(RAG)과 LangChain을 활용한 실습을 단계별로 제공합니다.  
ChromaDB를 벡터스토어로 사용하며, GitHub Codespaces 및 Jupyter Notebook 환경에서 실습할 수 있습니다.

---

## 폴더 구조

```
rag_demo/
├── app/                  # Streamlit 챗봇 앱
├── chat/                 # 챗봇 관련 모듈 및 코드
├── config/               # 환경설정 및 공통 설정 파일
├── data/                 # 샘플 문서 및 데이터
├── llm/                  # LLM 래퍼 및 프롬프트 관련 코드
├── notebook/             # 단계별 Jupyter 노트북
│   ├── 00_rag_intro_and_env.ipynb
│   ├── 01_data_loading_and_preprocessing.ipynb
│   ├── 02_embedding_and_chromadb.ipynb
│   ├── 03_retrieval_and_generation.ipynb
│   └── 04_advanced_evaluation_and_optimization.ipynb
├── chroma/               # ChromaDB 벡터스토어 저장 폴더 (자동 생성)
├── requirements.txt      # 필수 패키지 목록
├── .env.example          # 환경 변수 예시 파일
└── README.md
```

---

## 실습 모듈

1. **RAG/LangChain 이론 및 환경 준비**  
   - RAG 개념, LangChain 구조, 실습 환경 세팅  
   - → `notebook/00_rag_intro_and_env.ipynb`

2. **문서 데이터 로딩 및 전처리**  
   - 텍스트 파일 불러오기, 문서 분할, 클린징  
   - → `notebook/01_data_loading_and_preprocessing.ipynb`

3. **임베딩 생성 및 ChromaDB 벡터스토어 구축**  
   - OpenAI 임베딩 API 활용, ChromaDB에 저장  
   - → `notebook/02_embedding_and_chromadb.ipynb`

4. **RAG 체인: 검색 및 생성**  
   - 쿼리 임베딩, 유사도 검색, LLM 기반 답변 생성  
   - → `notebook/03_retrieval_and_generation.ipynb`

5. **(심화) 평가 및 최적화**  
   - RAG 평가 지표, 프롬프트 개선, 파라미터 튜닝  
   - → `notebook/04_advanced_evaluation_and_optimization.ipynb`

---

## 주요 라이브러리

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [OpenAI](https://platform.openai.com/)
- [Streamlit](https://streamlit.io/)
- [tiktoken](https://github.com/openai/tiktoken)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- 기타: pandas 등

---

## 실습 환경

- GitHub Codespaces 권장 (로컬 환경도 가능)
- Jupyter Notebook 기반 단계별 실습
- Streamlit 챗봇 예제 포함 (`app/streamlit_app.py`)

---

## 시작하기

1. Codespace에서 requirements.txt 설치  
   ```bash
   pip install -r requirements.txt
   ```
2. `.env.example`을 복사해 `.env` 생성 후 OpenAI API 키 입력  
3. Jupyter로 각 모듈 노트북 실행  
4. (옵션) Streamlit 챗봇 실행  
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## 기타 안내

- 샘플 문서는 `data/sample_docs.txt`에 위치
- ChromaDB 벡터스토어는 `./chroma` 폴더에 자동 저장
- `.env` 파일은 절대 커밋 금지!
- `chat/`, `config/`, `llm/` 폴더는 실습 코드 구조화 및 확장성을 위해 분리되어 있습니다.
- 실습 중 궁금한 점은 강사에게 문의하세요.
