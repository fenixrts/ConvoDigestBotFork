from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from telethon import TelegramClient
from telethon.tl.types import User
from aiogram import Bot, Dispatcher
from datetime import datetime, timedelta
import re
from src.config.config import load_config


# Интерфейс Telegram клиента
class ITelegramClient(ABC):
    @abstractmethod
    async def get_messages(self, chat_id: str, day_offset: int = 7, limit: int = None) -> List[Dict]:
        pass


# Реализация через Telethon и Aiogram
class TelegramBotClient(ITelegramClient):
    def __init__(self, api_id: int, api_hash: str, phone: str, session_name: str, bot_token: str):
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.bot = Bot(token=bot_token)
        self.dispatcher = Dispatcher()
        self.phone = phone
        self.ignored_sender_ids = load_config().get('IGNORED_SENDER_IDS', [])

    async def get_messages(self, chat_id: str, day_offset: int = 7, limit: int = None) -> List[Dict]:
        since = datetime.now() - timedelta(days=day_offset)
        messages = []
        async with self.client:
            await self.client.start(phone=self.phone)
            entity = await self._find_entity(self.client, chat_id)
            async for msg in self.client.iter_messages(entity, offset_date=since, reverse=True, limit=limit):
                # if msg.date.replace(tzinfo=None) < since:
                #     break
                if not self._should_skip_message(msg):
                    document_name = ''
                    if hasattr(msg, 'document') and msg.document:
                        for attr in getattr(msg.document, 'attributes', []):
                            if hasattr(attr, 'file_name'):
                                document_name = attr.file_name
                                break
                    message_data = {
                        'id': msg.id,
                        'date': msg.date.isoformat(),
                        'username': self._get_formatted_username(msg.sender),
                        'reply_to': msg.reply_to_msg_id,
                        'text': getattr(msg, 'text', '') or '',
                        'caption': getattr(msg, 'caption', '') or '',
                        'photo': hasattr(msg, 'photo') and msg.photo is not None,
                        'document': hasattr(msg, 'document') and msg.document is not None,
                        'document_name': document_name,
                        'video': hasattr(msg, 'video') and msg.video is not None,
                        'voice': hasattr(msg, 'voice') and msg.voice is not None,
                        'media_type': type(msg.media).__name__ if getattr(msg, 'media', None) else '',
                        'links': re.findall(r'(https?://\S+)',
                                            (getattr(msg, 'text', '') or '') + (getattr(msg, 'caption', '') or ''))
                    }
                    messages.insert(0, message_data)
        return messages

    def _should_skip_message(self, msg) -> bool:
        if msg.sender_id in self.ignored_sender_ids:
            return True
        if msg.fwd_from:
            fwd_sender_id = (getattr(msg.fwd_from.from_id, 'user_id', None)
                             or getattr(msg.fwd_from.from_id, 'channel_id', None))
            if fwd_sender_id in self.ignored_sender_ids:
                return True
        text = (msg.text or "").strip()
        if not text:
            return True
        if text.startswith('/') or re.search(r'@\w*telegram\w*', text, re.IGNORECASE):
            return True
        return False

    def _get_formatted_username(self, sender: User) -> str:
        if sender and getattr(sender, 'username', None):
            return f"@{sender.username}"
        if sender and getattr(sender, 'first_name', None) and getattr(sender, 'last_name', None):
            return f"{sender.first_name} {sender.last_name}".strip()
        if sender and getattr(sender, 'first_name', None):
            return sender.first_name
        return "Unknown"

    async def _find_entity(self, client, chat_id_or_username):
        try:
            return await client.get_entity(chat_id_or_username)
        except (ValueError, TypeError):
            pass
        async for dialog in client.iter_dialogs():
            if (str(dialog.id) == str(chat_id_or_username)
                    or dialog.name == chat_id_or_username
                    or getattr(dialog.entity, 'username', None) == chat_id_or_username):
                return dialog.entity
        raise ValueError(f"Чат с id/username '{chat_id_or_username}' не найден среди ваших диалогов!")
