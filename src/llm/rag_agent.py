"""
LangChain агент для работы с RAG и создания summary.
Простая реализация по принципу KISS.
"""
from typing import List, Dict, Any
import logging
import asyncio
import concurrent.futures
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.tools import StructuredTool, tool
except ImportError:
    try:
        from langchain_core.agents import AgentExecutor, create_openai_tools_agent
        from langchain_core.tools import StructuredTool, tool
    except ImportError:
        from langchain.agents.agent import AgentExecutor
        from langchain.agents.openai_tools import create_openai_tools_agent
        from langchain.tools import StructuredTool, tool

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.llm.rag_service import IRAGService
from src.llm.prompts import SYSTEM_PROMPT
from src.config.schemas import LLMResponse, RESPONSE_FORMAT


class RAGSearchInput(BaseModel):
    query: str = Field(description="Поисковый запрос")
    top_k: int = Field(default=15, description="Количество возвращаемых результатов (максимум 30)")


class RAGSearchOutput(BaseModel):
    results: List[Dict] = Field(description="Список результатов поиска")


def _rag_search_func(query: str, top_k: int = 15, rag_service=None) -> Dict[str, Any]:
    """Функция поиска релевантных сообщений в базе знаний чата."""
    if rag_service is None:
        logging.error("RAG сервис не предоставлен")
        return {"results": []}
    
    try:
        top_k = min(top_k, 30)
        # Синхронный запуск асинхронного поиска
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если event loop уже запущен, используем ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(rag_service.get_query_embedding(query))
                    )
                    query_emb = future.result()
            else:
                query_emb = loop.run_until_complete(rag_service.get_query_embedding(query))
            
            results = rag_service.search(query_emb, top_k=top_k)
            logging.info(f"RAG поиск: '{query}' → {len(results)} результатов")
            return {"results": results}
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                query_emb = loop.run_until_complete(rag_service.get_query_embedding(query))
                results = rag_service.search(query_emb, top_k=top_k)
                logging.info(f"RAG поиск: '{query}' → {len(results)} результатов")
                return {"results": results}
            finally:
                loop.close()
    except Exception as e:
        logging.error(f"Ошибка RAG поиска: {e}")
        return {"results": []}


class RAGAgent:
    """Простой LangChain агент для работы с RAG."""
    
    def __init__(self, llm, rag_service: IRAGService):
        self.llm = llm
        self.rag_service = rag_service
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> None:
        """Простой подход без агента - используем прямое обращение к RAG."""
        # Сохраняем rag_service для прямого использования
        self.rag_tool_func = lambda query, top_k=15: _rag_search_func(query, top_k, self.rag_service)
        return None
    
    async def summarize(self, messages: List[Dict], query: str, top_k: int = 20) -> Dict[str, Any]:
        """🔥 УЛЬТИМАТИВНЫЙ АГЕНТ для создания связного summary событий недели."""
        try:
            # Добавляем сообщения в RAG
            await self.rag_service.add_messages(messages)
            logging.info(f"Добавлено {len(messages)} сообщений в RAG")
            
            # ШАГ 1: Полный анализ хронологии событий недели
            logging.info("🧠 Собираю ПОЛНУЮ хронологию событий недели")
            chronological_events = self._analyze_week_chronology(messages)
            
            # ШАГ 2: Извлекаем ключевые события с помощью RAG
            key_events = await self._extract_key_events(query)
            
            # ШАГ 3: Строим связи между событиями (threads, replies)
            event_threads = self._build_event_threads(chronological_events, key_events)
            
            # ШАГ 4: Создаем связное повествование
            narrative_context = self._create_narrative_context(event_threads, chronological_events)
            
            logging.info(f"📖 Создан narrative контекст из {len(event_threads)} тредов и {len(chronological_events)} событий")
            
            # ШАГ 5: Генерируем связный отчет с УЛЬТИМАТИВНЫМ промптом
            prompt = self._create_ultimate_prompt(query, narrative_context)
            
            # Вызываем LLM напрямую
            response = await self.llm.ainvoke(prompt)
            logging.info(f"Получен ответ от LLM: {type(response)}")
            
            # Извлекаем результат
            if hasattr(response, 'content'):
                content = response.content
                logging.info(f"Контент ответа: {content[:200]}...")
                result = self._extract_structured_info(content)
            elif hasattr(response, 'main_fragments'):
                # Это уже Pydantic объект LLMResponse
                logging.info("Получен объект LLMResponse, извлекаю данные")
                result = {
                    'main_fragments': response.main_fragments,
                    'failures_and_rage': response.failures_and_rage,
                    'topics_to_discuss': response.topics_to_discuss
                }
            else:
                content = str(response)
                logging.info(f"Строковый ответ: {content[:200]}...")
                result = self._extract_structured_info(content)
            
            logging.info(f"✅ Результат обработки: {result}")
            return result
                
        except Exception as e:
            logging.error(f"Ошибка в RAG агенте: {e}")
            return self._create_fallback_response(query)
    
    def _extract_structured_info(self, output: Any) -> Dict[str, Any]:
        """Извлекает структурированную информацию из вывода агента."""
        try:
            logging.info(f"Обрабатываю output: {type(output)}")
            logging.info(f"Output содержимое: {str(output)[:500]}...")
            
            # Если результат уже в нужном формате, возвращаем как есть
            if isinstance(output, dict) and all(key in output for key in ['main_fragments', 'failures_and_rage', 'topics_to_discuss']):
                logging.info("Output уже в правильном формате")
                return output
            
            # Если это строка, пытаемся найти JSON
            if isinstance(output, str):
                import re
                import json
                
                logging.info("Пытаюсь парсить строку как JSON")
                
                # Сначала пытаемся парсить напрямую
                try:
                    parsed = json.loads(output.strip())
                    if isinstance(parsed, dict):
                        logging.info("Успешно распарсил JSON напрямую")
                        return parsed
                except json.JSONDecodeError:
                    logging.info("Прямой парсинг JSON не удался, ищу JSON блоки")
                
                # Ищем JSON блоки
                json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logging.info(f"Найден JSON блок: {json_str[:200]}...")
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        logging.info("Успешно распарсил JSON из блока")
                        return parsed
                
                # Пытаемся найти JSON без markdown
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logging.info(f"Найден JSON без markdown: {json_str[:200]}...")
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        logging.info("Успешно распарсил JSON без markdown")
                        return parsed
            
            # Если ничего не получилось, возвращаем fallback
            logging.warning("Не удалось извлечь структурированную информацию, возвращаю fallback")
            return self._create_fallback_response("анализ чата")
            
        except Exception as e:
            logging.error(f"Ошибка извлечения информации: {e}")
            return self._create_fallback_response("анализ чата")
    
    def _analyze_week_chronology(self, messages: List[Dict]) -> List[Dict]:
        """📅 Анализирует хронологию событий недели с сортировкой по времени."""
        from datetime import datetime
        
        # Сортируем сообщения по дате
        sorted_messages = []
        for msg in messages:
            try:
                if msg.get('date'):
                    # Парсим ISO дату
                    msg_date = datetime.fromisoformat(msg['date'].replace('Z', '+00:00'))
                    msg['parsed_date'] = msg_date
                    sorted_messages.append(msg)
            except Exception as e:
                logging.warning(f"Не удалось парсить дату: {msg.get('date')} - {e}")
                continue
        
        # Сортируем по времени
        sorted_messages.sort(key=lambda x: x['parsed_date'])
        
        logging.info(f"📅 Отсортировано {len(sorted_messages)} сообщений по хронологии")
        return sorted_messages
    
    async def _extract_key_events(self, query: str) -> List[Dict]:
        """🎯 Извлекаем ключевые события с помощью целевых RAG запросов."""
        key_event_queries = [
            # События с высокой эмоциональной окраской
            "ахуенно пиздец охереть заебись круто супер отлично",
            "факап косяк засада проебал обосрался провалил",
            "срач спор конфликт пригорело бомбит злит бесит",
            
            # Важные события и решения
            "решили обсудили договорились постановили выбрали",
            "новость событие произошло случилось стало известно",
            "проблема вопрос ситуация тема обсуждение",
            
            # Социальные события
            "день рождения праздник поздравляем отмечаем",
            "#повестка важно срочно внимание объявление",
            
            # Рабочие события
            "релиз выпуск деплой готово сделали завершили",
            "баг ошибка не работает сломалось упало",
            "дедлайн горит срочно успеть сдать время",
            
            # Персональные истории
            "рассказал поделился история случай произошло",

            "проблемы в отношениях, расстались",

            "дорогие квартиры, невозможно купить квартиру, жить негде",

            "трудности с детьми, ребёнок, пиздюк",

            f"{query}"  # Основной запрос
        ]
        
        all_key_events = []
        for event_query in key_event_queries:
            try:
                result = self.rag_tool_func(event_query, 8)
                if result and 'results' in result:
                    all_key_events.extend(result['results'])
            except Exception as e:
                logging.error(f"Ошибка извлечения ключевых событий для '{event_query}': {e}")
        
        # Убираем дубликаты
        seen_ids = set()
        unique_events = []
        for event in all_key_events:
            event_id = event.get('id', event.get('text', ''))
            if event_id not in seen_ids:
                seen_ids.add(event_id)
                unique_events.append(event)
        
        logging.info(f"🎯 Извлечено {len(unique_events)} ключевых событий")
        return unique_events
    
    def _build_event_threads(self, chronological_events: List[Dict], key_events: List[Dict]) -> List[Dict]:
        """🧵 Строим связи между событиями (треды, ответы)."""
        threads = []
        msg_by_id = {msg.get('id'): msg for msg in chronological_events if msg.get('id')}
        key_event_ids = {event.get('id') for event in key_events if event.get('id')}
        
        for event in key_events:
            thread = {
                'main_event': event,
                'replies': [],
                'context_before': [],
                'context_after': []
            }
            
            main_id = event.get('id')
            if not main_id:
                continue
            
            # Ищем ответы на это сообщение
            for msg in chronological_events:
                if msg.get('reply_to') == main_id:
                    thread['replies'].append(msg)
            
            # Ищем контекст до и после события
            main_event_time = event.get('parsed_date')
            if main_event_time:
                for msg in chronological_events:
                    msg_time = msg.get('parsed_date')
                    if msg_time and msg.get('id') != main_id:
                        time_diff = abs((msg_time - main_event_time).total_seconds())
                        
                        # Контекст в пределах 1 часа
                        if time_diff <= 3600:
                            if msg_time < main_event_time:
                                thread['context_before'].append(msg)
                            else:
                                thread['context_after'].append(msg)
            
            # Сортируем контекст по времени
            thread['context_before'].sort(key=lambda x: x.get('parsed_date', ''))
            thread['context_after'].sort(key=lambda x: x.get('parsed_date', ''))
            
            threads.append(thread)
        
        logging.info(f"🧵 Построено {len(threads)} тредов событий")
        return threads
    
    def _create_narrative_context(self, event_threads: List[Dict], chronological_events: List[Dict]) -> str:
        """📖 Создаем связное повествование для анализа."""
        from datetime import datetime
        
        narrative_parts = []
        
        # Группируем события по дням
        events_by_day = {}
        for event in chronological_events:
            try:
                event_date = event.get('parsed_date')
                if event_date:
                    day_key = event_date.strftime('%Y-%m-%d')
                    if day_key not in events_by_day:
                        events_by_day[day_key] = []
                    events_by_day[day_key].append(event)
            except:
                continue
        
        narrative_parts.append("=== ХРОНОЛОГИЯ СОБЫТИЙ НЕДЕЛИ ===\n")
        
        # Проходим по дням
        for day, day_events in sorted(events_by_day.items()):
            try:
                day_date = datetime.fromisoformat(day)
                day_name = day_date.strftime('%A, %d.%m.%Y')
                narrative_parts.append(f"\n📅 {day_name}:")
                
                # Добавляем события дня
                for event in day_events[:10]:  # Максимум 10 событий в день
                    username = event.get('username', 'Аноним')
                    text = event.get('text', '')
                    time = event.get('parsed_date').strftime('%H:%M') if event.get('parsed_date') else ''
                    
                    if text:
                        narrative_parts.append(f"  [{time}] {username}: {text}")
                
            except Exception as e:
                logging.warning(f"Ошибка обработки дня {day}: {e}")
                continue
        
        narrative_parts.append("\n\n=== КЛЮЧЕВЫЕ ТРЕДЫ И ОБСУЖДЕНИЯ ===\n")
        
        # Добавляем ключевые треды
        for i, thread in enumerate(event_threads[:10]):  # Топ-10 тредов
            main_event = thread['main_event']
            username = main_event.get('username', 'Аноним')
            text = main_event.get('text', '')
            
            narrative_parts.append(f"\n🔥 ТРЕД #{i+1}: {username}: {text}")
            
            # Добавляем ответы
            for reply in thread['replies'][:3]:  # Максимум 3 ответа
                reply_user = reply.get('username', 'Аноним')
                reply_text = reply.get('text', '')
                narrative_parts.append(f"   ↳ {reply_user}: {reply_text}")
            
            # Добавляем релевантный контекст
            if thread['context_before']:
                narrative_parts.append("   📝 Контекст до события:")
                for ctx in thread['context_before'][-2:]:  # Последние 2 сообщения до
                    ctx_user = ctx.get('username', 'Аноним')
                    ctx_text = ctx.get('text', '')
                    narrative_parts.append(f"      {ctx_user}: {ctx_text}")
        
        narrative = "\n".join(narrative_parts)
        
        # Ограничиваем размер
        if len(narrative) > 15000:
            narrative = narrative[:15000] + "\n\n[...контекст обрезан...]"
        
        logging.info(f"📖 Создан narrative контекст длиной {len(narrative)} символов")
        return narrative
    
    def _create_ultimate_prompt(self, query: str, narrative_context: str) -> str:
        """🔥 Создаем УЛЬТИМАТИВНЫЙ промпт для связного повествования."""
        
        ultimate_prompt = f"""{SYSTEM_PROMPT}

🎯 ЗАДАЧА: Создай СВЯЗНЫЙ ОТЧЕТ о событиях недели в чате, а не набор разрозненных фраз!

ТЕБЕ ДАНА ПОЛНАЯ ХРОНОЛОГИЯ СОБЫТИЙ НЕДЕЛИ:
{narrative_context}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"

🔥 ТРЕБОВАНИЯ К ОТЧЕТУ:

1. **СВЯЗНОСТЬ**: Пересказывай события как ИСТОРИЮ недели, показывая как одно вытекало из другого
2. **ХРОНОЛОГИЯ**: Упоминай последовательность событий ("сначала...", "потом...", "в итоге...")  
3. **ПЕРСОНАЖИ**: Следи за действиями конкретных людей на протяжении недели
4. **КОНТЕКСТ**: Объясняй ПОЧЕМУ что-то произошло, какие были предпосылки
5. **ЭМОЦИИ**: Передавай динамику настроений в чате

📝 СТРУКТУРА ОТВЕТА:

main_fragments - СВЯЗНОЕ повествование о главных событиях недели:
- "В понедельник @Username начал..., что привело к тому, что во вторник..."
- "Весь вторник чат бурлил из-за..., кульминацией стало..."
- "К середине недели ситуация с... переросла в..."

failures_and_rage - ИСТОРИИ о том, как всё шло по п***е:
- "Всё началось с того, что @Username..., а закончилось полным п***ецом когда..."
- "Эпичный факап случился, когда @User1 сказал..., а @User2 ответил..."

topics_to_discuss - ЧТО из обсуждений недели требует продолжения:
- "Незакрытая тема про..., которую @Username поднял в среду"
- "Спор между @User1 и @User2 о... так и не разрешился"

💡 ПРИМЕРЫ СВЯЗНОГО СТИЛЯ:
✅ "В понедельник @Vasya поднял тему о рефакторинге, что вызвало бурное обсуждение. @Petya сразу возразил, мол это будет залупа. К среде спор перерос в срач, когда @Ivan подключился со своими 5 копейками..."

❌ "@Vasya говорил о рефакторинге", "@Petya был против", "@Ivan что-то добавлял"

Создай отчет в формате JSON, каждый пункт - связная история, а не отдельные факты!"""

        return ultimate_prompt
    
    def _create_fallback_response(self, query: str) -> Dict[str, Any]:
        """Создает fallback ответ при ошибках."""
        return {
            'main_fragments': [f"Анализ по запросу '{query}' не удался"],
            'failures_and_rage': ["Технические проблемы с анализом"],
            'topics_to_discuss': ["Необходимо проверить работу системы"]
        }
