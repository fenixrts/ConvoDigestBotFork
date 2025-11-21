"""
Модуль для отправки итогового отчёта в Telegram-чат через aiogram (Bot API).
"""
import logging
from abc import ABC, abstractmethod
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramNotFound
from datetime import datetime

from src.config.config import load_config


# Интерфейс отправителя отчётов
class IReportSender(ABC):
    @abstractmethod
    async def send_report(self, report_json: dict, chat_id: str):
        pass


class TelegramReportSender(IReportSender):
    # Telegram message length limit
    MAX_MESSAGE_LENGTH = 4096
    
    def __init__(self, config=None):
        self.config = config or load_config()
        self.bot_token = self.config['BOT_TOKEN']
        self.bot = Bot(token=self.bot_token)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def escape_html(text: str) -> str:
        return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))

    def format_report(self, report_json: dict) -> str:
        parts = []
        today = datetime.now().strftime('%d.%m.%Y')
        parts.append(f"<b>💬 Итоги чата за неделю — {today}</b>\n")
        if report_json.get('main_fragments'):
            parts.append("📊 <b>Что обсуждали активнее всего:</b>")
            for i, item in enumerate(report_json['main_fragments'], start=1):
                parts.append(f"{i}. {self.escape_html(item)}")
            parts.append("")
        if report_json.get('failures_and_rage'):
            parts.append("💥 <b>О наболевшем:</b>")
            for i, item in enumerate(report_json['failures_and_rage'], start=1):
                parts.append(f"{i}. {self.escape_html(item)}")
            parts.append("")
        if report_json.get('topics_to_discuss'):
            parts.append("🦨 <b>Темы для нытинга:</b>")
            for i, item in enumerate(report_json['topics_to_discuss'], start=1):
                parts.append(f"{i}. {self.escape_html(item)}")
            parts.append("")
        hashtags = ' '.join(f"#{hashtag}" for hashtag in self.config.get('HASHTAGS', []))
        parts.append(f"{hashtags}\n")
        return '\n'.join(parts)
    
    def split_message(self, text: str) -> list[str]:
        """Split message into chunks that fit Telegram's length limit."""
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return [text]
        
        # Split by sections first (try to keep sections together)
        sections = text.split('\n\n')
        messages = []
        current_message = ""
        
        for section in sections:
            # If adding this section would exceed limit, start new message
            if current_message and len(current_message + '\n\n' + section) > self.MAX_MESSAGE_LENGTH:
                if current_message:
                    messages.append(current_message.strip())
                current_message = section
            else:
                if current_message:
                    current_message += '\n\n' + section
                else:
                    current_message = section
            
            # If single section is too long, split it by lines
            if len(current_message) > self.MAX_MESSAGE_LENGTH:
                if current_message != section:
                    # Save what we had before this section
                    messages.append(current_message.replace('\n\n' + section, '').strip())
                
                # Split the long section by lines
                lines = section.split('\n')
                temp_message = ""
                for line in lines:
                    if temp_message and len(temp_message + '\n' + line) > self.MAX_MESSAGE_LENGTH:
                        messages.append(temp_message.strip())
                        temp_message = line
                    else:
                        if temp_message:
                            temp_message += '\n' + line
                        else:
                            temp_message = line
                current_message = temp_message
        
        if current_message:
            messages.append(current_message.strip())
        
        return messages

    async def send_report(self, report_json: dict, chat_id: str):
        text = self.format_report(report_json)
        messages = self.split_message(text)
        
        self.logger.info(f"Sending report to chat_id: {chat_id} in {len(messages)} part(s)")
        
        try:
            for i, message in enumerate(messages, 1):
                self.logger.info(f"Sending part {i}/{len(messages)} (length: {len(message)})")
                await self.bot.send_message(chat_id, message, parse_mode=ParseMode.HTML)
                
                # Small delay between messages to avoid rate limiting
                if i < len(messages):
                    import asyncio
                    await asyncio.sleep(0.5)
            
            self.logger.info(f"Report successfully sent to chat_id: {chat_id}")
            
        except TelegramBadRequest as e:
            self.logger.error(f"Bad request when sending to chat_id {chat_id}: {e}")
            raise
        except TelegramForbiddenError as e:
            self.logger.error(f"Bot forbidden to send message to chat_id {chat_id}: {e}")
            raise
        except TelegramNotFound as e:
            self.logger.error(f"Chat not found for chat_id {chat_id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error sending message to chat_id {chat_id}: {e}")
            raise
        finally:
            # Close the bot session to prevent unclosed session warnings
            await self.bot.session.close()
