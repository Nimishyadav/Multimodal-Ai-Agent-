"""Choose a compatible embedding profile and generate vectors without mixing indexes."""
import os
import numpy as np
import requests

class CloudProcessingError(ValueError):
    pass


def cloud_mode():
    return os.getenv('RAG_PROCESSING_MODE', 'local').strip().lower() == 'cloud'


def embedding_profile():
    provider = os.getenv('RAG_EMBEDDING_PROVIDER', 'gemini').strip().lower()
    if provider == 'gemini':
        return {'provider':'gemini', 'model':'gemini-embedding-001', 'dimensions':768}
    if provider != 'local' or cloud_mode():
        raise CloudProcessingError('Cloud processing requires RAG_EMBEDDING_PROVIDER=gemini.')
    return {'provider':'local', 'model':'BAAI/bge-small-en-v1.5', 'dimensions':384}


def profile_key():
    profile = embedding_profile()
    return 'hybrid_' + profile['provider'] + '_' + str(profile['dimensions'])


def cloud_embeddings(texts):
    from app.config.settings import configured_key
    key = configured_key('GOOGLE_API_KEY', 'GEMINI_API_KEY')
    if not key:
        raise CloudProcessingError('Cloud embeddings need GOOGLE_API_KEY in local .env.')
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    try:
        rows=[]
        for start in range(0,len(texts),8):
            batch=texts[start:start+8]
            payload={'requests':[{'model':'models/gemini-embedding-001',
                     'content':{'parts':[{'text':text}]},'outputDimensionality':768,
                     'taskType':'SEMANTIC_SIMILARITY'} for text in batch]}
            response=requests.post('https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents',
                headers={'x-goog-api-key':key},json=payload,timeout=20)
            response.raise_for_status()
            vectors=response.json()['embeddings']
            if len(vectors)!=len(batch):
                raise ValueError('Missing embedding rows')
            rows.extend(item['values'] for item in vectors)
        values=np.asarray(rows,dtype=np.float32)
        if values.shape!=(len(texts),768) or not np.isfinite(values).all():
            raise ValueError('Invalid vector shape')
        lengths=np.linalg.norm(values,axis=1,keepdims=True)
        if np.any(lengths==0):raise ValueError('Empty vector')
        return values/lengths
    except Exception:
        raise CloudProcessingError('Cloud embeddings unavailable; check Google key, quota or connection. No CPU embedding fallback was started.') from None
