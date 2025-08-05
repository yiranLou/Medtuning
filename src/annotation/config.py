"""Mistral API配置"""
import os
from typing import Optional
from pydantic import BaseModel, Field, SecretStr


class MistralConfig(BaseModel):
    """Mistral API配置"""
    api_key: SecretStr = Field(..., description="Mistral API密钥")
    base_url: str = Field(
        default="https://api.mistral.ai/v1",
        description="API基础URL"
    )
    model: str = Field(
        default="mistral-large-latest",
        description="使用的模型"
    )
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")
    timeout: float = Field(default=300.0, gt=0, description="请求超时时间（秒）")
    temperature: float = Field(default=0.1, ge=0, le=1, description="生成温度")
    max_tokens: int = Field(default=4096, gt=0, description="最大生成token数")
    
    @classmethod
    def from_env(cls) -> 'MistralConfig':
        """从环境变量加载配置"""
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("未设置MISTRAL_API_KEY环境变量")
        
        return cls(
            api_key=SecretStr(api_key),
            base_url=os.getenv('MISTRAL_BASE_URL', cls.model_fields['base_url'].default),
            model=os.getenv('MISTRAL_MODEL', cls.model_fields['model'].default),
            max_retries=int(os.getenv('MISTRAL_MAX_RETRIES', cls.model_fields['max_retries'].default)),
            timeout=float(os.getenv('MISTRAL_TIMEOUT', cls.model_fields['timeout'].default)),
            temperature=float(os.getenv('MISTRAL_TEMPERATURE', cls.model_fields['temperature'].default)),
            max_tokens=int(os.getenv('MISTRAL_MAX_TOKENS', cls.model_fields['max_tokens'].default))
        )