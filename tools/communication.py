# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class phone_call(BaseTool):
    """Make a phone call."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        number = parameters.get("number", "")
        dial_call = parameters.get("dial_call", False)
        if dial_call:
            return ToolResponse(text=f"正在通过拨号界面拨打 {number}..."), 0.0, {}
        else:
            return ToolResponse(text=f"正在直接呼叫 {number}..."), 0.0, {}


class phone_sms(BaseTool):
    """Send an SMS message."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        recipient = parameters.get("sms_recipient", "")
        message = parameters.get("sms_message", "")
        return ToolResponse(text=f"已向 {recipient} 发送短信：{message}"), 0.0, {}


class send_email(BaseTool):
    """Send an email."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        to = parameters.get("to", "")
        subject = parameters.get("subject", "")
        body = parameters.get("body", "")
        return ToolResponse(text=f"已向 {to} 发送电子邮件，主题：{subject}，正文：{body}"), 0.0, {}


class create_contact(BaseTool):
    """Create a contact."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        first_name = parameters.get("first_name", "")
        last_name = parameters.get("last_name", "")
        phone = parameters.get("phone_number", "")
        email = parameters.get("email", "")
        name = f"{first_name} {last_name}".strip()
        return ToolResponse(text=f"已创建联系人 {name}（电话：{phone}，邮箱：{email}）。"), 0.0, {}


class search_contacts(BaseTool):
    """Search contacts."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        return ToolResponse(text=f'搜索联系人结果：找到与 "{query}" 相关的联系人。'), 0.0, {}
