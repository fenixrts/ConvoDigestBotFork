from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import re
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.rag_service import IRAGService
from src.llm.rag_agent import RAGAgent
from langchain_core.exceptions import OutputParserException


# Интерфейс LLM
class ILLM(ABC):
    @abstractmethod
    async def ainvoke(self, messages: List[Any]) -> Any:
        pass


# LLMService теперь принимает ILLM и IRAGService
class LLMService:
    def __init__(self, llm: ILLM, rag_service: IRAGService, use_agent: bool = True):
        self.llm = llm
        self.rag_service = rag_service
        self.use_agent = use_agent
        if use_agent:
            self.rag_agent = RAGAgent(llm, rag_service)

    async def summarize(self, messages: List[Dict], query: str, top_k: int = 20) -> Dict:
        if self.use_agent and hasattr(self, 'rag_agent'):
            logging.info("Использую простого RAG агента для создания summary")
            return await self.rag_agent.summarize(messages, query, top_k)
        else:
            logging.info("Использую стандартный RAG подход")
            return await self._standard_summarize(messages, query, top_k)
    
    async def _standard_summarize(self, messages: List[Dict], query: str, top_k: int = 20) -> Dict:
        """Стандартный RAG подход без агента."""
        await self.rag_service.add_messages(messages)
        query_emb = await self.rag_service.get_query_embedding(query)
        relevant_messages = self.rag_service.search(query_emb, top_k=top_k)
        if not relevant_messages:
            error_message = "Нет релевантных сообщений для анализа"
            logging.error(f"Ошибка в RAG summarize: {error_message}")
            raise Exception(error_message)
        payload = "\n".join([f"{msg.get('username', 'Anonymous')}: {msg.get('text', '')}" for msg in relevant_messages])
        return await self.llm_call(SYSTEM_PROMPT, payload)

    async def llm_call(self, prompt: str, content: str):
        messages = [SystemMessage(content=prompt), HumanMessage(content=content)]
        try:
            response = await self.llm.ainvoke(messages)
            return response.dict() if hasattr(response, 'dict') else response
        except OutputParserException as e:
            logging.error(f"OutputParserException: {e}")
            text = getattr(e, 'llm_output', str(e))
            try:
                text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
                return json.loads(text)
            except json.JSONDecodeError as e2:
                logging.error(f"JSON parse failed: {e2}")
                raise Exception(f"Не удалось получить валидный JSON: {e2}")
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise Exception(f"Не удалось получить валидный JSON: {e}")
