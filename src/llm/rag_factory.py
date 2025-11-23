"""
Фабрика для создания клиентов эмбеддингов (OpenAI/Ollama).
"""
import logging
from openai import AsyncOpenAI
from langchain_ollama import OllamaEmbeddings
from typing import Union, Any
import asyncio

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Универсальный клиент для работы с эмбеддингами.
    Поддерживает как OpenAI-совместимые API, так и Ollama через langchain.
    """
    def __init__(self, client: Union[AsyncOpenAI, OllamaEmbeddings], model: str, provider: str):
        self.client = client
        self.model = model
        self.provider = provider
        self._is_ollama = provider == 'ollama' and isinstance(client, OllamaEmbeddings)
    
    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Создает эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов для эмбеддинга
            
        Returns:
            Список векторов эмбеддингов
        """
        if self._is_ollama:
            # OllamaEmbeddings работает синхронно, используем asyncio.to_thread
            embeddings = await asyncio.to_thread(self.client.embed_documents, texts)
            return embeddings
        else:
            # OpenAI-совместимый API
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
    
    async def create_embedding(self, text: str) -> list[float]:
        """
        Создает эмбеддинг для одного текста.
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        if self._is_ollama:
            # OllamaEmbeddings работает синхронно, используем asyncio.to_thread
            embedding = await asyncio.to_thread(self.client.embed_query, text)
            return embedding
        else:
            # OpenAI-совместимый API
            response = await self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding


class RAGFactory:
    """Фабрика для создания клиентов эмбеддингов."""
    
    @staticmethod
    def create(rag_config: dict[str, Any]) -> EmbeddingClient:
        """
        Создает клиент для работы с эмбеддингами.
        
        Args:
            rag_config: Конфигурация RAG с полями:
                - provider: 'openai' или 'ollama'
                - model: название модели
                - base_url: базовый URL API (опционально)
                - api_key: API ключ (опционально для Ollama)
        
        Returns:
            EmbeddingClient для работы с эмбеддингами
        """
        provider = rag_config.get('provider', 'openai')
        model = rag_config.get('model', 'text-embedding-3-small')
        base_url = rag_config.get('base_url')
        api_key = rag_config.get('api_key', '')
        
        logger.info(f"Creating RAG client with provider: {provider}, model: {model}, base_url: {base_url}")
        
        if provider == 'openai':
            # OpenAI или OpenAI-совместимый API
            client = AsyncOpenAI(
                api_key=api_key or 'not-needed',
                base_url=base_url
            )
            return EmbeddingClient(client, model, provider)
        
        elif provider == 'ollama':
            # Используем OllamaEmbeddings из langchain-ollama
            # Если указан base_url, используем его, иначе по умолчанию http://localhost:11434
            ollama_base_url = base_url or 'http://localhost:11434'
            # Убираем /v1 если есть, так как OllamaEmbeddings сам добавляет путь
            if ollama_base_url.endswith('/v1'):
                ollama_base_url = ollama_base_url[:-3]
            
            client = OllamaEmbeddings(
                model=model,
                base_url=ollama_base_url
            )
            return EmbeddingClient(client, model, provider)
        
        else:
            raise ValueError(f"Unknown RAG provider: {provider}. Supported providers: 'openai', 'ollama'")

