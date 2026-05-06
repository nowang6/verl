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


class play_music(BaseTool):
    """Play music or songs."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        song = parameters.get("song_name", "")
        artist = parameters.get("artist", "")
        if song and artist:
            return ToolResponse(text=f"正在播放 {artist} 的歌曲《{song}》。"), 0.0, {}
        elif song:
            return ToolResponse(text=f"正在播放歌曲《{song}》。"), 0.0, {}
        else:
            return ToolResponse(text="正在播放音乐。"), 0.0, {}


class pause_music(BaseTool):
    """Pause currently playing music."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text="音乐已暂停。"), 0.0, {}


class take_picture(BaseTool):
    """Take a picture with the camera."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text="照片已拍摄并保存到相册。"), 0.0, {}
