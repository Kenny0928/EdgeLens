"""開發板推論（K210 的 .kmodel）：模型在板子上跑，電腦端不載入。

runner 讀 Frame.device_result，只套用信心門檻與類別名稱，結果標 where="device"。
"""

from __future__ import annotations

from typing import ClassVar

from edgelens.inference.base import Backend, register_backend


@register_backend
class DeviceBackend(Backend):
    kind: ClassVar[str] = "device"
    display_name_key: ClassVar[str] = "backend.device.name"
    extensions: ClassVar[tuple[str, ...]] = (".kmodel",)
    requires: ClassVar[tuple[str, ...]] = ()
    input_mode = "device"
    priority: ClassVar[int] = 10

    def load(self) -> None:
        """電腦端沒有東西要載入（模型檔在開發板的 flash 裡，電腦上不一定有）。"""

    def describe(self) -> str:
        from edgelens.i18n import tr

        return tr("backend.device.desc")
