"""
Модуль для отправки итогового отчёта в Telegram-чат через aiogram (Bot API).
"""
from abc import ABC, abstractmethod
from aiogram import Bot
from aiogram.enums import ParseMode
from datetime import datetime

from src.config.config import load_config


# Интерфейс отправителя отчётов
class IReportSender(ABC):
    @abstractmethod
    async def send_report(self, report_json: dict, chat_id: str):
        pass


class TelegramReportSender(IReportSender):
    def __init__(self, config=None):
        self.config = config or load_config()
        self.bot_token = self.config['BOT_TOKEN']
        self.bot = Bot(token=self.bot_token)

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

    async def send_report(self, report_json: dict, chat_id: str):
        text = self.format_report(report_json)
        await self.bot.send_message('-1002551893104', text, parse_mode=ParseMode.HTML)
