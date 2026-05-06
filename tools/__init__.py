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
"""Mobile action tools for multi-turn RL training."""

from .apps import list_application, open_application
from .display import (
    decrease_brightness,
    get_current_brightness,
    increase_brightness,
    set_brightness,
    take_screenshot,
)
from .device import (
    battery_status,
    bluetooth_status,
    mute_volume,
    open_bluetooth_settings,
    open_wifi_settings,
    set_volume,
    toggle_bluetooth,
    toggle_dnd,
    toggle_mobile_data,
    toggle_wifi,
    turn_off_flashlight,
    turn_on_flashlight,
)
from .communication import (
    create_contact,
    phone_call,
    phone_sms,
    search_contacts,
    send_email,
)
from .media import (
    pause_music,
    play_music,
    take_picture,
)
from .navigation import (
    get_location,
    show_map,
)
from .schedule import (
    create_calendar_event,
    set_alarm,
    set_timer,
)
from .browser import (
    open_browser,
    search_web,
)
from .weather import get_weather

__all__ = [
    "list_application",
    "open_application",
    "set_brightness",
    "decrease_brightness",
    "increase_brightness",
    "get_current_brightness",
    "take_screenshot",
    "battery_status",
    "bluetooth_status",
    "open_bluetooth_settings",
    "toggle_bluetooth",
    "mute_volume",
    "open_wifi_settings",
    "set_volume",
    "toggle_dnd",
    "toggle_mobile_data",
    "toggle_wifi",
    "turn_off_flashlight",
    "turn_on_flashlight",
    "create_contact",
    "phone_call",
    "phone_sms",
    "search_contacts",
    "send_email",
    "pause_music",
    "play_music",
    "take_picture",
    "get_location",
    "show_map",
    "create_calendar_event",
    "set_alarm",
    "set_timer",
    "open_browser",
    "search_web",
    "get_weather",
]
