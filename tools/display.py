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

import random
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class _DisplayState:
    """Shared state for display-related tools."""
    brightness = 50


class set_brightness(BaseTool):
    """Set screen brightness to a specified level."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        level = parameters.get("level", "50")
        _DisplayState.brightness = int(level)
        return ToolResponse(text=f"屏幕亮度已设置为 {level}。"), 0.0, {}


class decrease_brightness(BaseTool):
    """Decrease screen brightness."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _DisplayState.brightness = max(0, _DisplayState.brightness - 10)
        return ToolResponse(text=f"屏幕亮度已降低至 {_DisplayState.brightness}。"), 0.0, {}


class increase_brightness(BaseTool):
    """Increase screen brightness."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _DisplayState.brightness = min(100, _DisplayState.brightness + 10)
        return ToolResponse(text=f"屏幕亮度已提高至 {_DisplayState.brightness}。"), 0.0, {}


class get_current_brightness(BaseTool):
    """Get the current screen brightness level."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text=f"当前屏幕亮度为 {_DisplayState.brightness}。"), 0.0, {}


class take_screenshot(BaseTool):
    """Take a screenshot of the current screen."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text="屏幕截图已保存。"), 0.0, {}
