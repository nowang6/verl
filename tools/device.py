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


class _DeviceState:
    bluetooth_on = False
    wifi_on = True
    mobile_data_on = True
    flashlight_on = False
    volume = 70
    dnd_on = False
    battery_level = random.randint(60, 95)


class battery_status(BaseTool):
    """Get device battery status."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text=f"当前电池电量为 {_DeviceState.battery_level}%。"), 0.0, {}


class bluetooth_status(BaseTool):
    """Get Bluetooth status information."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        status = "已开启" if _DeviceState.bluetooth_on else "已关闭"
        return ToolResponse(text=f"蓝牙当前状态：{status}。"), 0.0, {}


class open_bluetooth_settings(BaseTool):
    """Open Bluetooth settings page."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text="正在打开蓝牙设置页面..."), 0.0, {}


class toggle_bluetooth(BaseTool):
    """Turn Bluetooth on or off."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        enable = parameters.get("enable", False)
        _DeviceState.bluetooth_on = bool(enable)
        status = "已开启" if _DeviceState.bluetooth_on else "已关闭"
        return ToolResponse(text=f"蓝牙{status}。"), 0.0, {}


class open_wifi_settings(BaseTool):
    """Open Wi-Fi settings."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text="正在打开Wi-Fi设置页面..."), 0.0, {}


class toggle_wifi(BaseTool):
    """Turn Wi-Fi on or off."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        enable = parameters.get("enable", False)
        _DeviceState.wifi_on = bool(enable)
        status = "已开启" if _DeviceState.wifi_on else "已关闭"
        return ToolResponse(text=f"Wi-Fi{status}。"), 0.0, {}


class toggle_mobile_data(BaseTool):
    """Turn mobile data on or off."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        enable = parameters.get("enable", False)
        _DeviceState.mobile_data_on = bool(enable)
        status = "已开启" if _DeviceState.mobile_data_on else "已关闭"
        return ToolResponse(text=f"移动数据{status}。"), 0.0, {}


class turn_on_flashlight(BaseTool):
    """Turn on the flashlight."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _DeviceState.flashlight_on = True
        return ToolResponse(text="手电筒已打开。"), 0.0, {}


class turn_off_flashlight(BaseTool):
    """Turn off the flashlight."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _DeviceState.flashlight_on = False
        return ToolResponse(text="手电筒已关闭。"), 0.0, {}


class set_volume(BaseTool):
    """Set device volume level."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        level = parameters.get("level", "70")
        _DeviceState.volume = int(level)
        return ToolResponse(text=f"音量已设置为 {level}。"), 0.0, {}


class mute_volume(BaseTool):
    """Mute the device."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _DeviceState.volume = 0
        return ToolResponse(text="设备已设为静音。"), 0.0, {}


class toggle_dnd(BaseTool):
    """Turn Do Not Disturb mode on or off."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        enable = parameters.get("enable", False)
        _DeviceState.dnd_on = bool(enable)
        status = "已开启" if _DeviceState.dnd_on else "已关闭"
        return ToolResponse(text=f"勿扰模式{status}。"), 0.0, {}
