from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from src.config.schemas import LLMResponse

class LLMFactory:
    @staticmethod
    def create(llm_config):
        provider = llm_config['provider']
        model_name = llm_config['model']
        base_url = llm_config.get('base_url')
        api_key = llm_config.get('api_key')
        output_parser = JsonOutputParser(pydantic_object=LLMResponse)

        if provider == 'openai':
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                temperature=0.2,
            )
            try:
                return llm.with_structured_output(LLMResponse, method="json_schema")
            except Exception:
                return llm | output_parser
        elif provider == 'ollama':
            llm = ChatOllama(
                base_url=base_url,
                model=model_name,
                temperature=0.2,
            )
            return llm | output_parser
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
