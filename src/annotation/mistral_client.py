"""Mistral Document AI客户端"""
import httpx
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import base64
from PIL import Image
import io

from .config import MistralConfig
from ..core.schemas import (
    DocumentAnnotation,
    BBoxAnnotation,
    get_document_schema_for_mistral,
    get_bbox_schema_for_mistral
)

logger = logging.getLogger(__name__)


class MistralAPIError(Exception):
    """Mistral API错误"""
    pass


class MistralClient:
    """Mistral Document AI客户端"""
    
    def __init__(self, config: Optional[MistralConfig] = None):
        self.config = config or MistralConfig.from_env()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout
        )
    
    def _log_retry(self, retry_state):
        """记录重试信息"""
        logger.warning(
            f"重试 {retry_state.attempt_number}/{self.config.max_retries}: "
            f"{retry_state.outcome.exception()}"
        )
    
    @retry(
        stop=stop_after_attempt(lambda self: self.config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((httpx.TimeoutException, MistralAPIError)),
        before=lambda retry_state: retry_state.fn.__self__._log_retry(retry_state)
    )
    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        files: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """发送API请求"""
        try:
            if files:
                # 多部分表单请求
                response = await self.client.post(
                    endpoint,
                    data=data,
                    files=files
                )
            else:
                # JSON请求
                response = await self.client.post(
                    endpoint,
                    json=data
                )
            
            if response.status_code != 200:
                error_msg = f"API错误 {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise MistralAPIError(error_msg)
            
            return response.json()
            
        except httpx.TimeoutException as e:
            logger.error(f"请求超时: {e}")
            raise
        except Exception as e:
            logger.error(f"请求失败: {e}")
            raise MistralAPIError(f"请求失败: {e}")
    
    def _prepare_image_for_api(self, image: Union[str, Path, Image.Image]) -> str:
        """准备图像用于API调用"""
        if isinstance(image, (str, Path)):
            # 从文件读取
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 获取MIME类型
            suffix = image_path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg'
            }.get(suffix, 'image/png')
            
        elif isinstance(image, Image.Image):
            # PIL图像
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            mime_type = 'image/png'
        else:
            raise ValueError("不支持的图像类型")
        
        # Base64编码
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_image}"
    
    async def annotate_document(
        self,
        pdf_path: Path,
        schema: Optional[Dict[str, Any]] = None,
        additional_instructions: str = ""
    ) -> DocumentAnnotation:
        """标注文档级信息"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 使用默认Schema
        if schema is None:
            schema = get_document_schema_for_mistral()
        
        # 构建提示
        prompt = self._build_document_prompt(schema, additional_instructions)
        
        # 准备请求数据
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的学术文献分析助手。请严格按照提供的JSON Schema格式提取文档信息。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "file",
                        "file": {
                            "content": base64.b64encode(pdf_content).decode('utf-8'),
                            "mime_type": "application/pdf"
                        }
                    }
                ]
            }
        ]
        
        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        # 发送请求
        response = await self._make_request("/chat/completions", request_data)
        
        # 解析响应
        try:
            content = response['choices'][0]['message']['content']
            data = json.loads(content)
            
            # 验证并返回
            return DocumentAnnotation.model_validate(data)
            
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"解析响应失败: {e}")
            raise MistralAPIError(f"无法解析API响应: {e}")
    
    async def annotate_bbox(
        self,
        crop_image: Union[str, Path, Image.Image],
        page_image: Optional[Union[str, Path, Image.Image]] = None,
        bbox_coords: Optional[List[int]] = None,
        anchor_text: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        additional_instructions: str = ""
    ) -> BBoxAnnotation:
        """标注边界框级信息"""
        # 使用默认Schema
        if schema is None:
            schema = get_bbox_schema_for_mistral()
        
        # 准备图像
        crop_image_data = self._prepare_image_for_api(crop_image)
        
        # 构建提示
        prompt = self._build_bbox_prompt(
            schema, 
            bbox_coords, 
            anchor_text, 
            additional_instructions
        )
        
        # 构建消息
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": crop_image_data}}
        ]
        
        # 如果提供了页面图像，也加入
        if page_image:
            page_image_data = self._prepare_image_for_api(page_image)
            content.append({
                "type": "image_url", 
                "image_url": {"url": page_image_data}
            })
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的图表分析助手。请严格按照提供的JSON Schema格式提取图表信息。"
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        # 发送请求
        response = await self._make_request("/chat/completions", request_data)
        
        # 解析响应
        try:
            content = response['choices'][0]['message']['content']
            data = json.loads(content)
            
            # 验证并返回
            return BBoxAnnotation.model_validate(data)
            
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"解析响应失败: {e}")
            raise MistralAPIError(f"无法解析API响应: {e}")
    
    def _build_document_prompt(
        self, 
        schema: Dict[str, Any], 
        additional_instructions: str
    ) -> str:
        """构建文档标注提示"""
        prompt = f"""请分析这篇学术论文，并按照以下JSON Schema格式提取结构化信息：

```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

注意事项：
1. 严格遵循Schema定义，不要添加额外字段
2. 所有文本字段去除控制字符和多余空白
3. 标题不要包含尾随标点
4. 关键词转为小写并去重，最多10个
5. 章节内容要合并同级段落
6. 空数组字段设为null而不是[]
7. DOI格式必须符合正则: ^10\\.\\d{{4,9}}/[-._;()\\/:a-zA-Z0-9]+$
8. 日期格式: YYYY-MM-DD 或 YYYY-MM 或 YYYY

{additional_instructions}

请直接返回JSON格式的结果。"""
        
        return prompt
    
    def _build_bbox_prompt(
        self,
        schema: Dict[str, Any],
        bbox_coords: Optional[List[int]],
        anchor_text: Optional[str],
        additional_instructions: str
    ) -> str:
        """构建边界框标注提示"""
        prompt = f"""请分析这个图表，并按照以下JSON Schema格式提取结构化信息：

```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

"""
        
        if bbox_coords:
            prompt += f"图表在页面中的位置坐标: {bbox_coords}\n"
        
        if anchor_text:
            prompt += f"图表附近的文本（用于参考）:\n{anchor_text}\n"
        
        prompt += """
注意事项：
1. 严格遵循Schema定义，不要添加额外字段
2. figure_type必须是: figure/table/equation/diagram/flowchart/other之一
3. caption要去除"Figure 1:"等编号前缀
4. 单位必须标准化为SI单位（如mL, μg, mmHg等）
5. key_findings必须是可直接观察到的，不能包含推断性词汇
6. 表格类型才能有table_csv字段
7. 所有坐标必须是整数

{additional_instructions}

请直接返回JSON格式的结果。"""
        
        return prompt
    
    async def annotate_batch(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Union[DocumentAnnotation, BBoxAnnotation, Exception]]:
        """批量标注"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_task(task):
            async with semaphore:
                try:
                    task_type = task.get('type')
                    if task_type == 'document':
                        return await self.annotate_document(**task['params'])
                    elif task_type == 'bbox':
                        return await self.annotate_bbox(**task['params'])
                    else:
                        raise ValueError(f"未知任务类型: {task_type}")
                except Exception as e:
                    logger.error(f"任务失败: {e}")
                    return e
        
        results = await asyncio.gather(
            *[process_task(task) for task in tasks],
            return_exceptions=True
        )
        
        return results
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()