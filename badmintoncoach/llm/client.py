"""OpenAI兼容的教练LLM客户端"""
import json
from typing import Optional

from openai import AsyncOpenAI

from ..config import LLMConfig
from .prompts import SYSTEM_PROMPT, build_coaching_prompt


class CoachLLM:
    """OpenAI兼容的教练LLM客户端"""

    def __init__(self, config: LLMConfig):
        if not config.api_key:
            raise ValueError("LLM API Key 未配置，请在 config.yaml 或设置页面配置")
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def get_coaching(self, analysis_summary: dict) -> str:
        """从分析结果获取教练建议
        Args:
            analysis_summary: report.json内容
        Returns:
            LLM生成的教练建议（JSON字符串）
        """
        user_prompt = build_coaching_prompt(analysis_summary)

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def chat(self, message: str, context: Optional[dict] = None) -> str:
        """自由对话模式（用户追问教练）"""
        system = SYSTEM_PROMPT
        if context:
            system += f"\n\n当前分析上下文:\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
