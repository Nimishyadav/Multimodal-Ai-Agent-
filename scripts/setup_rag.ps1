# Run from the project directory using an installed Python 3.11+ interpreter.
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath '.venv-standard\Scripts\python.exe') {
    & .\.venv-standard\Scripts\python.exe -m venv .venv-rag
} elseif (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    & .\.venv\Scripts\python.exe -m venv .venv-rag
} else {
    py -3.13 -m venv .venv-rag
}
if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated RAG environment.' }
& .\.venv-rag\Scripts\python.exe -m pip install -r requirements-rag.txt
if ($LASTEXITCODE -ne 0) { throw 'Optional RAG dependencies could not be installed.' }
& .\.venv-rag\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Resolve the reported dependency conflicts before using multimodal RAG.' }
Write-Output 'Runtime installed. Set DOCUMENT_RAG_BACKEND=raganything in .env, then restart Streamlit.'
Write-Output 'MinerU and local embeddings download model weights on first use. Office conversion may need LibreOffice.'
