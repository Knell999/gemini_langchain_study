# Gemini API와 LangChain 학습 프로젝트

이 레포지토리는 Google Gemini API를 활용한 LangChain 학습을 위한 프로젝트입니다. LangChain의 핵심 개념부터 고급 기능까지 단계별로 학습할 수 있도록 구성되어 있습니다.

## 📁 프로젝트 구조

```
gemini_langchain_study/
├── gemini.ipynb          # 메인 학습 노트북
├── my_document.txt      # RAG 시스템 테스트용 문서
├── pyproject.toml       # 프로젝트 설정 파일
├── README.md           # 프로젝트 문서
└── chroma_db/          # Chroma 벡터 데이터베이스 저장소
    ├── chroma.sqlite3
    └── e3addcf5-f247-4e95-a672-db991d9f3557/
```

## 🚀 주요 학습 내용

### 1. 기본 설정 및 환경 구성
- Google Gemini API 키 설정
- LangChain과 Gemini 모델 초기화
- 환경 변수 관리

### 2. LangChain 기본 사용법
- **단일 질문 처리**: 기본적인 질문-답변 시스템
- **대화 기록 관리**: 멀티턴 대화 구현
- **임베딩 생성**: 텍스트의 벡터 표현 생성

### 3. 함수 호출 (Function Calling)
- **도구 정의**: Pydantic 모델을 활용한 도구 스키마 정의
- **도구 바인딩**: Gemini 모델에 도구 연결
- **도구 호출 처리**: 자동 함수 실행 및 결과 처리

### 4. LangChain 체인 (Chains)
- **기본 체인**: 프롬프트 → LLM → 파서의 기본 파이프라인
- **순차 체인**: 다단계 작업을 위한 체인 연결
- **라우터 체인**: 조건부 로직으로 동적 체인 선택

### 5. RAG (Retrieval-Augmented Generation) 시스템
완전한 RAG 시스템을 단계별로 구축:

1. **문서 로드**: `TextLoader`를 사용한 파일 읽기
2. **텍스트 분할**: `RecursiveCharacterTextSplitter`로 청크 분할
3. **임베딩 생성**: Gemini 임베딩 모델로 벡터 변환
4. **벡터 저장소**: Chroma 데이터베이스에 임베딩 저장
5. **검색기 구성**: 유사성 기반 문서 검색

## 🛠️ 설치 및 실행

### 1. 필요한 패키지 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 필요한 패키지 설치
pip install langchain langchain-google-genai langchain-community chromadb python-dotenv
```

### 2. 환경 설정

프로젝트 루트에 `.env` 파일을 생성하고 Google API 키를 설정:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Google AI Studio에서 API 키 발급

1. [Google AI Studio](https://aistudio.google.com/)에 접속
2. API 키 생성 및 복사
3. `.env` 파일에 키 추가

### 4. 실행 방법

#### Jupyter 노트북으로 실행 (권장)
```bash
jupyter notebook gemini.ipynb
```

## 📚 학습 가이드

### 초급자를 위한 학습 순서
1. **기본 설정**: API 키 설정 및 모델 초기화
2. **단순 질문**: 기본적인 질문-답변 시스템 이해
3. **대화 관리**: 대화 기록을 활용한 멀티턴 대화
4. **체인 기초**: LCEL을 활용한 기본 체인 구성

### 중급자를 위한 학습 순서
1. **함수 호출**: 도구 정의 및 자동 함수 실행
2. **복합 체인**: 순차 체인과 라우터 체인 구현
3. **RAG 기초**: 문서 로드부터 검색까지의 기본 파이프라인

### 고급자를 위한 학습 순서
1. **RAG 최적화**: 청크 크기, 검색 매개변수 튜닝
2. **커스텀 체인**: 복잡한 비즈니스 로직을 위한 맞춤형 체인
3. **성능 최적화**: 비동기 처리, 스트리밍, 캐싱

## 🔍 주요 기능 상세

### LCEL (LangChain Expression Language)
```python
# 기본 체인 구성
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "LangChain"})
```

### RAG 시스템 구축
```python
# 문서 처리 파이프라인
loader = TextLoader("document.txt")
docs = loader.load()
chunks = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
```

### 함수 호출
```python
# 도구 정의 및 바인딩
@tool
def calculator(a: int, b: int) -> int:
    return a + b

llm_with_tools = llm.bind_tools([calculator])
```

## 🔧 사용된 기술 스택

- **LLM**: Google Gemini 2.5 Flash Preview
- **프레임워크**: LangChain
- **임베딩**: Google Generative AI Embeddings
- **벡터 데이터베이스**: Chroma
- **개발 환경**: Jupyter Notebook, Python 3.13+

## 📖 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [Google AI Studio](https://aistudio.google.com/)
- [Chroma 벡터 데이터베이스](https://www.trychroma.com/)