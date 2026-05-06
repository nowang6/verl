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


class list_application(BaseTool):
    """List installed applications on the phone."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        apps = [
            "电话", "短信", "相机", "相册", "设置", "时钟", "日历",
            "浏览器", "音乐", "文件管理", "计算器", "天气",
            "地图", "通讯录", "邮件", "手电筒",
        ]
        return ToolResponse(text=f"已安装的应用程序：{', '.join(apps)}"), 0.0, {}


class open_application(BaseTool):
    """Open an application on the phone."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        app_name = parameters.get("application_name", "")
        if not app_name:
            return ToolResponse(text="请指定要打开的应用程序名称。"), 0.0, {}
        return ToolResponse(text=f"正在打开 {app_name} 应用程序..."), 0.0, {}
