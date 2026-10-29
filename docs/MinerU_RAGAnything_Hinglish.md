# MinerU aur RAG-Anything kaise aur kyun add hua

## Pehle kya tha

PDF se pypdf plain text nikalta tha aur BM25 relevant text chunks dhoondta tha. Ye simple text PDFs ke liye useful hai. Lekin scan, figure, table layout aur equation ka structure plain text mein lose ho sakta hai. Multi-agent orchestration pehle se LangGraph mein hai; usko replace nahi kiya.

## Ab kya add hua

MinerU document ko parse karke structured content list deta hai. Har block ka type aur page metadata hota hai. RAG-Anything us list ko process karta hai aur LightRAG se graph/vector retrieval structures banwata hai. Existing LangChain text/vision models explain/extract karte hain. FastEmbed local vectors banata hai. UI mein wahi chat aur file uploader hai.

## Har stage ka role

1. Upload: file size/format check hota hai. Filename ke badle short internal disk names Windows path issues kam karte hain.
2. Parse: MinerU actual document se text, table, figure aur equation blocks identify karta hai. Page index ko correct one-based page references mein convert karte hain.
3. Understand: text/vision callbacks supported content describe karte hain. Figure description model-derived hoti hai; source check important hai.
4. Embed: BAAI/bge-small-en-v1.5 text ko 384-number vector mein convert karta hai. FastEmbed/ONNX local hai; embedding API key nahi chahiye. First use par weights download hote hain.
5. Graph: LightRAG entities aur relationships nikalta hai. Example: document explicitly kehta hai Upload -> Parse -> Retrieve, toh related concepts retrieval mein useful ho sakte hain. Graph original source ka replacement nahi hai.
6. Retrieve: normal question par hybrid graph/vector context plus source passages milte hain. Exact PDF page request par sirf requested page ka evidence use hota hai.
7. Answer: Document Agent tool result dekhkar explain karta hai. Orchestrator baaki required specialists coordinate karta hai.

## Excel aur image ka difference

Excel ka exact total ab bhi full loaded table par Pandas calculate karta hai. RAG mein sirf table metadata aur first 10 rows ka preview jaata hai; us preview se complete-file total infer karna galat hai. Standalone images ka direct vision path bhi available hai. Har image mein OCR text hona zaroori nahi; no-text image ki direct vision answer aa sakti hai chahe knowledge index incomplete ho.

## Kaunsi library kya karti hai

MinerU = parser/OCR/layout extraction. RAG-Anything = multimodal content ingestion/query framework. LightRAG = graph/vector retrieval engine. FastEmbed = local embedding inference. LangChain = model and tool integration. LangGraph = orchestrator/state/tool loops. Pandas = exact local calculations. Streamlit = interface. In sab ka role alag hai.

## Separate environment kyun

MinerU/RAG-Anything ke Torch, ONNX aur document packages heavy hain. Inhe .venv-rag mein install kiya, taaki current app ka .venv-standard environment disturb na ho. App JSON requests ke through worker process ko call karta hai. API credentials command-line arguments ya request JSON mein nahi jaate; worker local configuration use karta hai.

## Fail hua toh kya

Parser fail ho aur ordinary readable text available ho toh local reader fallback ho sakta hai. Parser successful lekin graph build fail ho toh parsed_only status dikhata hai; source text still usable hai. Sidebar se incomplete indexing retry kar sakte hain. Query fail ho toh local evidence return hota hai. Har provider unavailable ho toh generated answer guarantee nahi hai.

## Important practical points

First run slow ho sakta hai because model weights download hote hain. Index build bhi LLM calls leta hai; free quota consume ho sakta hai. Office files ke wrapper conversion ko LibreOffice chahiye ho sakta hai; current machine par LibreOffice detect nahi hua, isliye Office multimodal parsing ka claim nahi kiya. PDFs/table/figure sample ka actual end-to-end run successful hua.

Multimodal mode original source aur derived knowledge disk par data/multimodal_rag mein store karta hai. Ye old RAM-only behavior se different hai. Clearing active chat scope stored files automatically delete nahi karta. data/ ko share mat karo. Cross-chat automatic knowledge discovery enable nahi ki.

## Run

Current installation mein dependencies installed hain aur DOCUMENT_RAG_BACKEND=raganything enabled hai. Server restart karo, phir current app interpreter se Streamlit run karo. New installation ke liye scripts/setup_rag.ps1 aur detailed English setup guide follow karo.

## Verified

105 regression tests ran; 2 optional external-sample tests skipped. Plain-text PDF aur table/figure PDF real MinerU/RAG-Anything pipeline se verify hue. Streamlit startup: zero exceptions. Exact-page isolation, graph context, media path checks aur failure fallback tests included hain. Har format/layout/provider ke liye guarantee nahi hai.
