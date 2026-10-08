"""右側面板外掛：discover_panels() 自動找出 panels 底下所有 Panel 子類別。"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import traceback

from edgelens.gui.panels.base import Panel

log = logging.getLogger(__name__)

PACKAGE = "edgelens.gui.panels"


def discover_panels(package: str = PACKAGE) -> list[type[Panel]]:
    """匯入 package 底下所有模組，回傳有 panel_id 的 Panel 子類別，依 (order, panel_id) 排序。

    某個模組匯入失敗（例如缺選配套件）時只記日誌並略過，其他面板照常載入。
    """
    from edgelens.core.registry import ImportProblem, import_problems

    pkg = importlib.import_module(package)
    found: dict[str, type[Panel]] = {}
    for info in pkgutil.iter_modules(pkg.__path__, prefix=package + "."):
        leaf = info.name.rsplit(".", 1)[-1]
        if leaf == "base" or leaf.startswith("_"):
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001 - 單一面板壞掉不能讓程式打不開
            log.warning("面板模組 %s 匯入失敗：%s", info.name, e)
            import_problems.append(ImportProblem(info.name, repr(e), traceback.format_exc()))
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, Panel) and cls is not Panel and cls.panel_id and cls.__module__ == mod.__name__:
                if cls.panel_id in found:
                    log.warning("面板 ID %s 重複（%s），略過", cls.panel_id, cls.__module__)
                    continue
                found[cls.panel_id] = cls
    return sorted(found.values(), key=lambda c: (c.order, c.panel_id))
