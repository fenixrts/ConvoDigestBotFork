import asyncio
import logging
from openai import AsyncOpenAI

from src.config.config import load_config
from src.llm.llm_factory import LLMFactory
from src.llm.llm_service import LLMService
from src.llm.rag_service import FaissRAGService
from src.telegram.telegram_service import TelegramBotClient
from src.telegram.telegram_report_sender import TelegramReportSender
from src.scheduler.scheduler_service import AioScheduler
from src.scheduler.main_pipeline import DigestPipeline

config = load_config()
MODE = config['MODE']

llm = LLMFactory.create(config['LLM_CONFIG'])
rag = FaissRAGService(config['RAG_CONFIG'])
llm_service = LLMService(llm, rag)

telegram = TelegramBotClient(
    api_id=config['TELEGRAM_API_ID'],
    api_hash=config['TELEGRAM_API_HASH'],
    phone=config['TELEGRAM_PHONE'],
    session_name=config['TELEGRAM_SESSION_NAME'],
    bot_token=config['BOT_TOKEN']
)

report_sender = TelegramReportSender(config)
scheduler = AioScheduler()
pipeline = DigestPipeline(llm_service, telegram, report_sender, scheduler)

CHAT_ID = config['TELEGRAM_CHAT_ID']
DIST_CHAT_ID = config['TELEGRAM_DIST_CHAT_ID']
QUERY = config['RAG_CONFIG']['query']
CRON = {'day_of_week': 'sun', 'hour': 18, 'minute': 0}


async def main():
    await pipeline.run_digest(CHAT_ID, DIST_CHAT_ID, QUERY)
    pipeline.schedule_digest(CHAT_ID, DIST_CHAT_ID, QUERY, CRON, job_id='weekly_digest')
    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    asyncio.run(main())
