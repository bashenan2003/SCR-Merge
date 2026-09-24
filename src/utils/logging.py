"""Structured logging with checkpoint markers."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Logs pipeline progress with module-level markers."""

    def __init__(self, name: str = "SCR-Merge", log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        self.logger.addHandler(console)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def module_start(self, module_name: str) -> None:
        self.logger.info("=" * 50)
        self.logger.info(f"[START] {module_name}")
        self.logger.info("=" * 50)

    def module_end(self, module_name: str, elapsed: float) -> None:
        self.logger.info(f"[END] {module_name} ({elapsed:.2f}s)")

    def checkpoint(self, name: str) -> None:
        self.logger.info(f"[CHECKPOINT] {name}")

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def metric(self, name: str, value) -> None:
        self.logger.info(f"[METRIC] {name} = {value}")


_default_logger: Optional[PipelineLogger] = None


def get_logger(log_file: Optional[Path] = None) -> PipelineLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = PipelineLogger(log_file=log_file)
    return _default_logger
