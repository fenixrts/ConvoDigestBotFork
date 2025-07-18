from src.llm.llm_service import LLMService
from src.telegram.telegram_service import ITelegramClient
from src.telegram.telegram_report_sender import IReportSender
from src.scheduler.scheduler_service import IScheduler


class DigestPipeline:
    def __init__(
            self,
            llm_service: LLMService,
            telegram: ITelegramClient,
            report_sender: IReportSender,
            scheduler: IScheduler
    ):
        self.llm_service = llm_service
        self.telegram = telegram
        self.report_sender = report_sender
        self.scheduler = scheduler

    async def run_digest(self, chat_id_from: str, chat_id_to: str, query: str, top_k: int = 20):
        messages = await self.telegram.get_messages(chat_id_from)
        summary = await self.llm_service.summarize(messages, query, top_k)
        await self.report_sender.send_report(summary, chat_id_to)

    def schedule_digest(self, chat_id_from: str, chat_id_to: str, query: str, cron: dict, job_id: str = None,
                        top_k: int = 20):
        async def job():
            await self.run_digest(chat_id_from, chat_id_to, query, top_k)

        self.scheduler.add_cron_job(job, cron, job_id)

    def shutdown(self):
        self.scheduler.shutdown()
