import asyncio
import logging
import argparse
import sys

from src.config.config import load_config
from src.llm.llm_factory import LLMFactory
from src.llm.llm_service import LLMService
from src.llm.rag_service import FaissRAGService
from src.telegram.telegram_service import TelegramBotClient
from src.telegram.telegram_report_sender import TelegramReportSender
from src.scheduler.scheduler_service import AioScheduler
from src.scheduler.main_pipeline import DigestPipeline
from src.telegram.bot import run_bot

logger = logging.getLogger(__name__)

config = load_config()
CHAT_ID = config['TELEGRAM_CHAT_ID']
DIST_CHAT_ID = config['TELEGRAM_DIST_CHAT_ID']
QUERY = config['RAG_CONFIG']['query']
CRON = {'day_of_week': 'sun', 'hour': 18, 'minute': 0}


def create_pipeline():
    """Создает и возвращает настроенный пайплайн."""
    llm = LLMFactory.create(config['LLM_CONFIG'])
    rag = FaissRAGService(config['RAG_CONFIG'])
    llm_service = LLMService(llm, rag, use_agent=config.get('USE_AGENT', False))

    telegram = TelegramBotClient(
        api_id=config['TELEGRAM_API_ID'],
        api_hash=config['TELEGRAM_API_HASH'],
        phone=config['TELEGRAM_PHONE'],
        session_name=config['TELEGRAM_SESSION_NAME'],
        bot_token=config['BOT_TOKEN']
    )

    report_sender = TelegramReportSender(config)
    scheduler = AioScheduler()
    return DigestPipeline(llm_service, telegram, report_sender, scheduler)


async def run_manual_digest():
    """Запускает дайджест один раз без планировщика."""
    logger.info("Запуск ручного дайджеста...")
    pipeline = create_pipeline()
    try:
        await pipeline.run_digest(CHAT_ID, DIST_CHAT_ID, QUERY)
        logger.info("Ручной дайджест успешно завершен")
    except Exception as e:
        logger.error(f"Ошибка при выполнении ручного дайджеста: {e}", exc_info=True)
        sys.exit(1)


async def run_scheduler():
    """Запускает планировщик с автоматическим выполнением по расписанию."""
    logger.info("Запуск планировщика...")
    pipeline = create_pipeline()
    
    # Запускаем планировщик
    scheduler = pipeline.scheduler
    await scheduler.start()
    
    # Выполняем дайджест сразу при запуске
    await pipeline.run_digest(CHAT_ID, DIST_CHAT_ID, QUERY)
    
    # Настраиваем расписание
    pipeline.schedule_digest(CHAT_ID, DIST_CHAT_ID, QUERY, CRON, job_id='weekly_digest')
    logger.info(f"Планировщик настроен: дайджест будет выполняться каждое воскресенье в {CRON['hour']}:{CRON['minute']:02d}")
    
    # Ждем бесконечно
    await asyncio.Event().wait()


async def run_bot_only():
    """Запускает только Telegram-бота без планировщика."""
    logger.info("Запуск Telegram-бота...")
    await run_bot()


async def run_both():
    """Запускает и бота, и планировщик одновременно."""
    logger.info("Запуск бота и планировщика...")
    pipeline = create_pipeline()
    
    # Запускаем планировщик
    scheduler = pipeline.scheduler
    await scheduler.start()
    
    # Выполняем дайджест сразу при запуске
    await pipeline.run_digest(CHAT_ID, DIST_CHAT_ID, QUERY)
    
    # Настраиваем расписание
    pipeline.schedule_digest(CHAT_ID, DIST_CHAT_ID, QUERY, CRON, job_id='weekly_digest')
    logger.info(f"Планировщик настроен: дайджест будет выполняться каждое воскресенье в {CRON['hour']}:{CRON['minute']:02d}")
    
    # Запускаем бота в фоне, передавая pipeline для команды /run_digest
    bot_task = asyncio.create_task(run_bot(pipeline))
    
    # Ждем завершения (не должно произойти)
    await bot_task


def main():
    parser = argparse.ArgumentParser(
        description='ConvoDigestBot - генератор дайджестов Telegram-чатов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Режимы работы:
  manual     - Запустить дайджест один раз и завершить работу
  scheduler  - Запустить планировщик (автоматический запуск по расписанию)
  bot        - Запустить только Telegram-бота с командами
  both       - Запустить и бота, и планировщик одновременно (по умолчанию)

Примеры:
  python main.py                    # Запуск в режиме 'both' (из конфига MODE)
  python main.py manual             # Одноразовый запуск дайджеста
  python main.py scheduler          # Только планировщик
  python main.py bot                # Только бот
  python main.py both               # Бот + планировщик
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['manual', 'scheduler', 'bot', 'both'],
        default=None,
        help='Режим работы (если не указан, используется MODE из конфига)'
    )
    
    args = parser.parse_args()
    
    # Определяем режим работы
    mode = args.mode or config.get('MODE', 'both').lower()
    
    if mode not in ['manual', 'scheduler', 'bot', 'both']:
        logger.warning(f"Неизвестный режим '{mode}', используется 'both'")
        mode = 'both'
    
    logger.info(f"Запуск в режиме: {mode}")
    
    # Запускаем соответствующий режим
    if mode == 'manual':
        asyncio.run(run_manual_digest())
    elif mode == 'scheduler':
        asyncio.run(run_scheduler())
    elif mode == 'bot':
        asyncio.run(run_bot_only())
    elif mode == 'both':
        asyncio.run(run_both())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    main()
