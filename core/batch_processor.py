"""
批量处理器：提供简单的批量操作抽象（占位实现）。

此模块在项目中并未广泛使用，但为了保持完整性，提供一个小型、可扩展的实现。
"""

from typing import List, Callable, Optional
import os


class BatchProcessor:
    """用于执行批处理任务的简单帮助类。"""

    def __init__(self):
        self._running = False

    def process(self, items: List[str], worker: Callable[[str], None], progress_callback: Optional[Callable[[int, int], None]] = None):
        """对 items 列表逐项调用 worker(item)。

        progress_callback(current_index, total) 可选。
        """
        total = len(items)
        self._running = True

        for idx, item in enumerate(items, start=1):
            if not self._running:
                break
            try:
                worker(item)
            except Exception:
                # 失败项忽略，调用者可在 worker 内处理异常
                pass

            if progress_callback:
                try:
                    progress_callback(idx, total)
                except Exception:
                    pass

        self._running = False

    def stop(self):
        """请求停止当前批处理（非立即中断）。"""
        self._running = False


__all__ = ["BatchProcessor"]
