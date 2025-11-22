import asyncio
import logging
from telethon import TelegramClient
from src.config.config import load_config

logger = logging.getLogger(__name__)

config = load_config()
API_ID = int(config['TELEGRAM_API_ID'])
API_HASH = config['TELEGRAM_API_HASH']
PHONE = config['TELEGRAM_PHONE']
SESSION_NAME = config['TELEGRAM_SESSION_NAME']

async def create_telethon_session():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.connect()
    if not await client.is_user_authorized():
        logger.info("[Telethon] Необходима авторизация. Введите код, который придёт в Telegram...")
        await client.start(phone=PHONE)
        logger.info("[Telethon] Авторизация завершена!")
    else:
        logger.info("[Telethon] Уже авторизован.")
    # Получаем user_id владельца сессии
    me = await client.get_me()
    logger.info(f"[Telethon] Ваш user_id: {me.id}")
    await client.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    asyncio.run(create_telethon_session()) 