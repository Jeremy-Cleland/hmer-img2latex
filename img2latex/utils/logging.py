"""
Logging utilities for the img2latex project.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Dictionary to keep track of loggers that have been created
_LOGGERS: Dict[str, logging.Logger] = {}
_GLOBAL_FILE_HANDLER = None


class ImmediateFileHandler(logging.FileHandler):
    """Custom FileHandler that flushes after each emit to ensure logs are written immediately."""

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        # Don't use the parent's initialization, manage our own file
        logging.Handler.__init__(self)
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.terminator = "\n"

        # Open the file immediately with line buffering (buffering=1)
        self._open()

    def _open(self):
        """Open the file with line buffering"""
        self.stream = open(
            self.baseFilename, self.mode, buffering=1, encoding=self.encoding
        )
        return self.stream

    def emit(self, record):
        """Override emit method to flush immediately after writing and ensure stream is open."""
        if self.stream is None:
            self._open()
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        """
        Close the file and clean up.
        """
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
    log_level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    timestamp: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    Args:
        name: Name of the logger (typically __name__)
        log_level: Log level (e.g. INFO, DEBUG)
        log_file: Optional filename (e.g. 'train.log')
        log_dir: Directory to store log file
        timestamp: If True, append timestamp to log_file
        use_colors: Use colored log output for console

    Returns:
        logging.Logger instance
    """
    global _GLOBAL_FILE_HANDLER

    if name in _LOGGERS:
        return _LOGGERS[name]

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Disable propagation for all loggers except the root logger
    # This is the key fix for double logging
    logger.propagate = False

    # Clear old handlers
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    if use_colors and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(log_format, date_format))
    else:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - Handle both global and per-logger file handlers
    if log_dir and log_file:
        os.makedirs(log_dir, exist_ok=True)
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{Path(log_file).stem}_{timestamp_str}{Path(log_file).suffix}"
        file_path = os.path.join(log_dir, log_file)

        # Use global file handler for main logger and its children
        handler_to_use = None
        if name == "img2latex" or name.startswith("img2latex."):
            # Create a new file handler if not already present
            if (
                _GLOBAL_FILE_HANDLER is None
                or _GLOBAL_FILE_HANDLER.baseFilename != file_path
            ):
                # Close previous handler if it exists
                if _GLOBAL_FILE_HANDLER is not None:
                    _GLOBAL_FILE_HANDLER.close()

                # Use our custom ImmediateFileHandler instead of regular FileHandler
                _GLOBAL_FILE_HANDLER = ImmediateFileHandler(file_path, mode="a")
                _GLOBAL_FILE_HANDLER.setFormatter(formatter)
                _GLOBAL_FILE_HANDLER.setLevel(log_level)
            handler_to_use = _GLOBAL_FILE_HANDLER
        # For non-img2latex loggers that need their own file handler
        else:
            # Create a new file handler for this logger using our custom handler
            file_handler = ImmediateFileHandler(file_path, mode="a")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            handler_to_use = file_handler

        # Ensure the handler is attached (it might have been removed)
        # Add handler_to_use only if it's valid and not already present
        if handler_to_use and handler_to_use not in logger.handlers:
            logger.addHandler(handler_to_use)
            # Avoid logging this message repeatedly if handler already exists
            # Initialize a flag on the handler itself the first time it's used for logging path
            if not getattr(handler_to_use, "_logging_path_logged", False):
                logger.info(f"Logging to file: {file_path}")
                handler_to_use._logging_path_logged = True  # Set flag after logging

    _LOGGERS[name] = logger
    return logger


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def configure_logging(cfg) -> logging.Logger:
    """
    Global logging setup using config.

    Args:
        cfg: Config object from configuration

    Returns:
        Root logger
    """
    global _GLOBAL_FILE_HANDLER
    _GLOBAL_FILE_HANDLER = None

    # Force Python to use unbuffered mode - this affects stdout/stderr
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Pull settings from config
    experiment_name = (
        cfg.training.experiment_name
        if hasattr(cfg, "training") and hasattr(cfg.training, "experiment_name")
        else "default"
    )
    log_level = cfg.logging.level if hasattr(cfg, "logging") else "INFO"
    use_colors = cfg.logging.use_colors if hasattr(cfg, "logging") else True
    log_to_file = cfg.logging.log_to_file if hasattr(cfg, "logging") else True

    # Check if we have a path manager instance
    try:
        from img2latex.utils.path_utils import path_manager

        # Use the path manager to get the experiment directory
        log_dir = path_manager.get_log_dir(experiment_name)
    except (ImportError, AttributeError):
        # Fallback if path_manager isn't available
        from pathlib import Path

        output_dir = (
            cfg.training.output_dir
            if hasattr(cfg, "training") and hasattr(cfg.training, "output_dir")
            else "outputs"
        )
        log_dir = Path(output_dir) / experiment_name / "logs"
        os.makedirs(log_dir, exist_ok=True)

    # Create log filename based on command
    log_file = "train.log"
    if (
        hasattr(cfg, "logging")
        and hasattr(cfg.logging, "log_file")
        and cfg.logging.log_file
    ):
        log_file = cfg.logging.log_file

    # Silence noisy libraries
    for noisy in ["matplotlib", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Configure the root logger
    root_root_logger = logging.getLogger()
    root_root_logger.setLevel(
        logging.WARNING
    )  # Only show warnings and above for third-party libs
    # Remove all handlers from the root logger to avoid duplicated logs
    for handler in root_root_logger.handlers[:]:
        root_root_logger.removeHandler(handler)

    # Create main app logger
    if log_to_file:
        root_logger = get_logger(
            "img2latex",
            log_level=log_level,
            log_dir=str(log_dir),
            log_file=log_file,
            use_colors=use_colors,
        )
    else:
        root_logger = get_logger(
            "img2latex",
            log_level=log_level,
            use_colors=use_colors,
        )

    # Set up module loggers
    for module in [
        "data",
        "model",
        "training",
        "evaluation",
        "utils",
    ]:
        get_logger(f"img2latex.{module}", log_level=log_level)

    # Register cleanup
    import atexit

    def flush_all_loggers():
        """Flush all loggers at exit to ensure logs are written."""
        for _name, logger in _LOGGERS.items():
            for handler in logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

    atexit.register(flush_all_loggers)

    return root_logger


def log_execution_params(logger, cfg):
    """
    Log core execution metadata for reproducibility.

    Args:
        logger: Logger instance
        cfg: Config object from configuration
    """
    logger.info("------ Execution Context ------")
    logger.info(f"Command        : {cfg.get('command', 'N/A')}")
    logger.info(f"Model          : {cfg.model.get('name', 'N/A')}")
    logger.info(f"Dataset        : {cfg.data.get('dataset_name', 'N/A')}")
    logger.info(f"Experiment     : {cfg.get('experiment_name', 'N/A')}")
    logger.info(f"Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-------------------------------")