"""
Модуль для хранения и поиска эмбеддингов сообщений (FAISS).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import faiss
import numpy as np
import logging
from src.llm.rag_factory import RAGFactory

logger = logging.getLogger(__name__)


# Интерфейс RAG
class IRAGService(ABC):
    @abstractmethod
    async def add_messages(self, messages: List[Dict]):
        pass

    @abstractmethod
    async def get_query_embedding(self, query: str) -> np.ndarray:
        pass

    @abstractmethod
    def search(self, query_emb: np.ndarray, top_k: Optional[int] = None) -> List[Dict]:
        pass


# Реализация на FAISS
class FaissRAGService(IRAGService):
    def __init__(self, rag_config):
        logger.info(f"{__name__} initialized with provider={rag_config.get('provider', 'openai')}, model={rag_config.get('model')}, base_url={rag_config.get('base_url')}")
        self.embedding_client = RAGFactory.create(rag_config)
        self.top_k = rag_config['top_k']
        self.index = None
        self.messages: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.dim = None

    async def add_messages(self, messages: List[Dict]):
        texts = [self._message_context(msg) for msg in messages]
        embeddings = await self.get_embeddings(texts)
        if not embeddings:
            return
        if self.index is None:
            self.dim = len(embeddings[0])
            self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.messages.extend(messages)
        self.embeddings.extend(embeddings)

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Получает эмбеддинги для списка текстов."""
        embeddings = await self.embedding_client.create_embeddings(texts)
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    async def get_query_embedding(self, query: str) -> np.ndarray:
        """Получает эмбеддинг для запроса."""
        embedding = await self.embedding_client.create_embedding(query)
        return np.array(embedding, dtype=np.float32)

    def search(self, query_emb: np.ndarray, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or self.top_k
        D, I = self.index.search(np.array([query_emb], dtype=np.float32), top_k)
        return [self.messages[i] for i in I[0] if i < len(self.messages)]

    def _message_context(self, msg: Dict) -> str:
        parts = []
        username = msg.get('username', 'Anonymous')
        text = msg.get('text', '')
        parts.append(f"{username}: {text}")
        if msg.get('photo'):
            caption = msg.get('caption', '')
            parts.append(f"[Фото] {caption}")
        if msg.get('document'):
            doc_name = msg.get('document_name', 'Документ')
            caption = msg.get('caption', '')
            parts.append(f"[Документ: {doc_name}] {caption}")
        if msg.get('video'):
            caption = msg.get('caption', '')
            parts.append(f"[Видео] {caption}")
        if msg.get('voice'):
            parts.append("[Голосовое сообщение]")
        if msg.get('links'):
            for link in msg['links']:
                parts.append(f"[Ссылка] {link}")
        if msg.get('media_type'):
            parts.append(f"[Вложение: {msg['media_type']}] {msg.get('caption', '')}")
        return " ".join(parts)
