import io
import json
import logging
import asyncio
from aiogram.types import BufferedInputFile, BotCommand
from src.config.config import load_config
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from src.telegram.telethon_client import get_list_chats
from src.scheduler.main_pipeline import DigestPipeline

logger = logging.getLogger(__name__)

# Описание команд для меню Telegram
BOT_COMMANDS = [
    BotCommand(command="start", description="Приветствие и краткая справка"),
    BotCommand(command="help", description="Подробная справка по боту"),
    BotCommand(command="run_digest", description="Запустить дайджест вручную"),
    BotCommand(command="list_chats_json", description="Получить список чатов в JSON"),
]


def create_pipeline():
    """Создает и возвращает настроенный пайплайн."""
    from src.llm.llm_factory import LLMFactory
    from src.llm.llm_service import LLMService
    from src.llm.rag_service import FaissRAGService
    from src.telegram.telegram_service import TelegramBotClient
    from src.telegram.telegram_report_sender import TelegramReportSender
    from src.scheduler.scheduler_service import AioScheduler
    
    config = load_config()
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


async def run_bot(pipeline: DigestPipeline = None):
    """
    Запускает Telegram-бота с командами.
    
    Args:
        pipeline: Опциональный пайплайн для запуска дайджеста. 
                  Если не передан, будет создан новый при необходимости.
    """
    bot = Bot(token=load_config().get('BOT_TOKEN'))
    dp = Dispatcher()
    config = load_config()
    OWNER_ID = config.get('TELEGRAM_OWNER_ID')
    CHAT_ID = config['TELEGRAM_CHAT_ID']
    DIST_CHAT_ID = config['TELEGRAM_DIST_CHAT_ID']
    QUERY = config['RAG_CONFIG']['query']

    def is_owner(message: types.Message) -> bool:
        if not OWNER_ID:
            return False
        return str(message.from_user.id) == str(OWNER_ID)

    # Устанавливаем команды для Telegram-клиента
    await bot.set_my_commands(BOT_COMMANDS)

    @dp.message(Command("start"))
    async def cmd_start(message: types.Message):
        if not is_owner(message):
            return
        await message.answer(
            "Привет! Я бот для генерации еженедельных дайджестов чата.\n\n"
            "Доступные команды:\n"
            "/help — подробная справка\n"
            "/run_digest — запустить дайджест вручную\n"
            "/list_chats_json — список чатов в JSON")

    @dp.message(Command("help"))
    async def cmd_help(message: types.Message):
        if not is_owner(message):
            return
        await message.answer(
            "Я автоматически собираю сообщения за неделю, генерирую отчёт и отправляю его в этот чат.\n\n"
            "Доступные команды:\n"
            "/start — приветствие\n"
            "/help — справка\n"
            "/run_digest — запустить дайджест вручную\n"
            "/list_chats_json — список чатов в JSON")

    @dp.message(Command("run_digest"))
    async def cmd_run_digest(message: types.Message):
        if not is_owner(message):
            return
        
        status_message = await message.answer("🔄 Запускаю дайджест... Это может занять несколько минут.")
        
        try:
            # Создаем пайплайн, если не передан
            current_pipeline = pipeline or create_pipeline()
            
            # Запускаем дайджест
            await current_pipeline.run_digest(CHAT_ID, DIST_CHAT_ID, QUERY)
            
            await status_message.edit_text("✅ Дайджест успешно сгенерирован и отправлен!")
            logger.info(f"Дайджест запущен вручную пользователем {message.from_user.id}")
            
        except Exception as e:
            error_msg = f"❌ Ошибка при генерации дайджеста: {str(e)}"
            logger.error(f"Ошибка при ручном запуске дайджеста: {e}", exc_info=True)
            await status_message.edit_text(error_msg)

    @dp.message(Command("list_chats_json"))
    async def cmd_list_chats_json(message: types.Message):
        if not is_owner(message):
            return
        temp_message = await message.answer("🔄 Генерирую отчёт по чатам, подожди пару секунд...")

        try:
            # Получаем список чатов
            chats = await get_list_chats()
            json_text = json.dumps(chats, ensure_ascii=False, indent=2)

            # Создаём буфер для JSON-файла
            buffer = io.BytesIO(json_text.encode("utf-8"))
            buffer.name = "chat_list.json"
            await message.answer_document(BufferedInputFile(buffer.read(), filename=buffer.name))

            await temp_message.delete()
        except Exception as e:
            logger.error(f"Ошибка при получении списка чатов: {e}", exc_info=True)
            await temp_message.edit_text(f"❌ Ошибка при получении списка чатов: {str(e)}")

    logger.info("Telegram-бот запущен и готов к работе")
    await dp.start_polling(bot)
