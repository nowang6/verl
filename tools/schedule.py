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


class create_calendar_event(BaseTool):
    """Create a new calendar event."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        title = parameters.get("title", "")
        dt = parameters.get("datetime", "")
        return ToolResponse(text=f'已创建日历事件："{title}"，时间：{dt}。'), 0.0, {}


class set_alarm(BaseTool):
    """Set an alarm."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        dt = parameters.get("datetime", "")
        label = parameters.get("label", "闹钟")
        repeat = parameters.get("repeat", "一次性")
        return ToolResponse(text=f"已设置闹钟：{label}，时间：{dt}，重复：{repeat}。"), 0.0, {}


class set_timer(BaseTool):
    """Set a countdown timer."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        duration = parameters.get("duration", "")
        label = parameters.get("label", "计时器")
        return ToolResponse(text=f"已设置倒计时：{label}，时长：{duration}。"), 0.0, {}
