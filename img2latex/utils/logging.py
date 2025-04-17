"""
Logging utilities for the img2latex project.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Track existing loggers and a shared file handler
_LOGGERS: Dict[str, logging.Logger] = {}
_GLOBAL_FILE_HANDLER = None


class ImmediateFileHandler(logging.FileHandler):
    """FileHandler that flushes after each emit for real‑time logs."""

    def __init__(self, filename: str, mode: str = "a", encoding=None, delay=False):
        logging.Handler.__init__(self)
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.terminator = "\n"
        self._open()

    def _open(self):
        self.stream = open(
            self.baseFilename, self.mode, buffering=1, encoding=self.encoding
        )
        return self.stream

    def emit(self, record: logging.LogRecord):
        if self.stream is None:
            self._open()
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self.acquire()
        try:
            if self.stream:
                self.flush()
                self.stream.close()
                self.stream = None
        finally:
            self.release()


def get_logger(
    name: str,
    log_level: Union[str, int],
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    timestamp: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    All parameters must be provided (no hidden defaults).
    """
    global _GLOBAL_FILE_HANDLER

    if name in _LOGGERS:
        return _LOGGERS[name]

    # Normalize log level
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, date_fmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    # If you want colored output, configure here based on use_colors
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler if requested
    if log_dir and log_file:
        os.makedirs(log_dir, exist_ok=True)
        filename = log_file
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem, suffix = Path(log_file).stem, Path(log_file).suffix
            filename = f"{stem}_{ts}{suffix}"
        path = os.path.join(log_dir, filename)

        # Reuse a global handler for the main logger
        if name == "img2latex" or name.startswith("img2latex."):
            if not _GLOBAL_FILE_HANDLER or _GLOBAL_FILE_HANDLER.baseFilename != path:
                if _GLOBAL_FILE_HANDLER:
                    _GLOBAL_FILE_HANDLER.close()
                _GLOBAL_FILE_HANDLER = ImmediateFileHandler(path, mode="a")
                _GLOBAL_FILE_HANDLER.setFormatter(formatter)
                _GLOBAL_FILE_HANDLER.setLevel(log_level)
            handler = _GLOBAL_FILE_HANDLER
        else:
            handler = ImmediateFileHandler(path, mode="a")
            handler.setFormatter(formatter)
            handler.setLevel(log_level)

        if handler not in logger.handlers:
            logger.addHandler(handler)
            logger.info(f"Logging to file: {path}")

    _LOGGERS[name] = logger
    return logger


def configure_logging(config: dict) -> logging.Logger:
    """
    Global logging setup driven entirely by the YAML config dict.

    Expects:
      config['training']['experiment_name']
      config['logging']['level']
      config['logging']['log_to_file']
      config['logging']['use_colors']
      config['logging']['log_file']
    """
    global _GLOBAL_FILE_HANDLER
    _GLOBAL_FILE_HANDLER = None

    os.environ["PYTHONUNBUFFERED"] = "1"

    # Pull required settings, else KeyError
    exp_name = config["training"]["experiment_name"]
    lg_level = config["logging"]["level"]
    use_colors = config["logging"]["use_colors"]
    to_file = config["logging"]["log_to_file"]
    log_file = config["logging"]["log_file"]

    # Determine log directory
    try:
        from img2latex.utils.path_utils import path_manager

        log_dir = path_manager.get_log_dir(exp_name)
    except (ImportError, AttributeError):
        out_dir = config["training"].get("output_dir", "outputs")
        log_dir = Path(out_dir) / exp_name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

    # Create root app logger
    if to_file:
        root_logger = get_logger(
            "img2latex",
            log_level=lg_level,
            log_dir=str(log_dir),
            log_file=log_file,
            timestamp=False,
            use_colors=use_colors,
        )
    else:
        root_logger = get_logger("img2latex", log_level=lg_level, use_colors=use_colors)

    # Configure submodule loggers (list defined in config if desired)
    # Example: modules = config['logging'].get('modules', [])
    # For now, keep the standard set
    for mod in ["data", "model", "training", "evaluation", "utils"]:
        get_logger(f"img2latex.{mod}", lg_level, use_colors=use_colors)

    # Silence noisy libs
    for noisy in ["matplotlib", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # At exit, flush all handlers
    import atexit

    def _flush_all():
        for logger in _LOGGERS.values():
            for h in logger.handlers:
                if hasattr(h, "flush"):
                    h.flush()

    atexit.register(_flush_all)
    return root_logger


def log_execution_params(logger: logging.Logger, config: dict):
    """
    Log core execution metadata.

    Expects:
      config.get('command', None)
      config['model']['name']
      config['training']['experiment_name']
    """
    logger.info("------ Execution Context ------")
    cmd = config.get("command", None)
    if cmd is not None:
        logger.info(f"Command        : {cmd}")
    logger.info(f"Model          : {config['model']['name']}")
    logger.info(f"Experiment     : {config['training']['experiment_name']}")
    logger.info(f"Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-------------------------------")
