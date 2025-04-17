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
        self.encoding = encoding or "utf-8"  # Default to UTF-8
        self.terminator = "\n"
        self.stream = None

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)

        if not delay:
            self._open()

    def _open(self):
        try:
            self.stream = open(
                self.baseFilename, self.mode, buffering=1, encoding=self.encoding
            )
            return self.stream
        except Exception as e:
            print(f"Error opening log file {self.baseFilename}: {e}")
            raise

    def emit(self, record: logging.LogRecord):
        if self.stream is None:
            self._open()
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception as e:
            print(f"Error writing to log file: {e}")
            self.handleError(record)

    def flush(self):
        if self.stream:
            try:
                self.stream.flush()
            except Exception as e:
                print(f"Error flushing log file: {e}")

    def close(self):
        self.acquire()
        try:
            if self.stream:
                self.flush()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"Error closing log file: {e}")
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

        # Debug logging path
        print(f"Setting up logging to file: {path}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Reuse a global handler for the main logger
        if name == "img2latex" or name.startswith("img2latex."):
            if not _GLOBAL_FILE_HANDLER or _GLOBAL_FILE_HANDLER.baseFilename != path:
                if _GLOBAL_FILE_HANDLER:
                    _GLOBAL_FILE_HANDLER.close()
                try:
                    _GLOBAL_FILE_HANDLER = ImmediateFileHandler(path, mode="a")
                    _GLOBAL_FILE_HANDLER.setFormatter(formatter)
                    _GLOBAL_FILE_HANDLER.setLevel(log_level)
                    print(f"Created file handler for {path}")
                except Exception as e:
                    print(f"Error creating file handler: {e}")
            handler = _GLOBAL_FILE_HANDLER
        else:
            try:
                handler = ImmediateFileHandler(path, mode="a")
                handler.setFormatter(formatter)
                handler.setLevel(log_level)
            except Exception as e:
                print(f"Error creating file handler: {e}")
                handler = None

        if handler and handler not in logger.handlers:
            logger.addHandler(handler)
            print(f"Added handler for {path} to logger {name}")
            # Test that we can write to the log
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
    try:
        exp_name = config["training"]["experiment_name"]
        lg_level = config["logging"].get("level", "INFO")
        use_colors = config["logging"].get("use_colors", True)
        to_file = config["logging"].get("log_to_file", True)
        log_file = config["logging"].get("log_file", "train.log")
    except KeyError as e:
        print(f"Missing required config for logging: {e}")
        # Use defaults
        exp_name = config.get("training", {}).get(
            "experiment_name", "default_experiment"
        )
        lg_level = "INFO"
        use_colors = True
        to_file = True
        log_file = "train.log"

    # Determine log directory
    try:
        from img2latex.utils.path_utils import path_manager

        log_dir = path_manager.get_log_dir(exp_name)
        print(f"Using log directory: {log_dir}")
    except (ImportError, AttributeError) as e:
        print(f"Error getting log directory from path_manager: {e}")
        out_dir = config.get("training", {}).get("output_dir", "outputs")
        log_dir = Path(out_dir) / exp_name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created fallback log directory: {log_dir}")

    # Create root app logger
    if to_file:
        try:
            print(f"Setting up file logging to {log_dir}/{log_file}")
            root_logger = get_logger(
                "img2latex",
                log_level=lg_level,
                log_dir=str(log_dir),
                log_file=log_file,
                timestamp=False,
                use_colors=use_colors,
            )
            # Test write to logger
            root_logger.info(f"Logging initialized for experiment: {exp_name}")
        except Exception as e:
            print(
                f"Error setting up file logger: {e}. Falling back to console-only logging."
            )
            root_logger = get_logger(
                "img2latex", log_level=lg_level, use_colors=use_colors
            )
    else:
        root_logger = get_logger("img2latex", log_level=lg_level, use_colors=use_colors)

    # Configure submodule loggers (list defined in config if desired)
    try:
        # Get list of modules to configure from config
        modules = config.get("logging", {}).get(
            "modules", ["data", "model", "training", "evaluation", "utils"]
        )

        for mod in modules:
            if to_file:
                get_logger(
                    f"img2latex.{mod}",
                    log_level=lg_level,
                    log_dir=str(log_dir),
                    log_file=log_file,
                    use_colors=use_colors,
                )
            else:
                get_logger(f"img2latex.{mod}", lg_level, use_colors=use_colors)
    except Exception as e:
        print(f"Error configuring module loggers: {e}")

    # Silence noisy libs
    for noisy in ["matplotlib", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # At exit, flush all handlers
    import atexit

    def _flush_all():
        try:
            for logger in _LOGGERS.values():
                for h in logger.handlers:
                    if hasattr(h, "flush"):
                        h.flush()
            print("Successfully flushed all log handlers at exit")
        except Exception as e:
            print(f"Error flushing log handlers at exit: {e}")

    atexit.register(_flush_all)

    print(
        f"Logging configured successfully. Root logger has {len(root_logger.handlers)} handlers"
    )

    # Log basic execution metadata
    try:
        log_execution_params(root_logger, config)
    except Exception as e:
        print(f"Error logging execution parameters: {e}")

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


def log_to_file(message: str, filepath: str, mode: str = "a"):
    """
    Log a message directly to a specific file.

    This is useful for adding logs outside the standard logging system,
    for example, to log debugging information or specific metrics.

    Args:
        message: Message to log
        filepath: Path to the log file
        mode: File opening mode (default 'a' for append)
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{timestamp} | {message}\n"

        # Write to file
        with open(filepath, mode) as f:
            f.write(full_message)
            f.flush()

        print(f"Logged message to file: {filepath}")
        return True
    except Exception as e:
        print(f"Error logging to file {filepath}: {e}")
        return False


def check_logger_status(logger_name: str = None):
    """
    Check and print the status of the logging system.

    This is a diagnostic tool to help debug logging issues.

    Args:
        logger_name: Specific logger to check, or None to check all loggers

    Returns:
        Dictionary with logger status information
    """
    loggers_info = {}

    if logger_name:
        loggers_to_check = [logger_name]
    else:
        loggers_to_check = list(_LOGGERS.keys())

    print(f"Checking status for {len(loggers_to_check)} loggers:")

    for name in loggers_to_check:
        if name in _LOGGERS:
            logger = _LOGGERS[name]
            handlers_info = []

            for i, handler in enumerate(logger.handlers):
                handler_type = type(handler).__name__

                # Get handler-specific info
                if hasattr(handler, "baseFilename"):
                    handler_info = {
                        "type": handler_type,
                        "file": handler.baseFilename,
                        "level": logging.getLevelName(handler.level),
                        "formatter": bool(handler.formatter),
                    }

                    # Check if file is writable
                    try:
                        if os.path.exists(handler.baseFilename):
                            # Check if directory is writable
                            dir_writable = os.access(
                                os.path.dirname(handler.baseFilename), os.W_OK
                            )
                            # Check if file is writable
                            file_writable = os.access(handler.baseFilename, os.W_OK)
                            handler_info["dir_writable"] = dir_writable
                            handler_info["file_writable"] = file_writable

                            if not dir_writable or not file_writable:
                                print(
                                    f"WARNING: {handler.baseFilename} or its directory is not writable"
                                )
                    except Exception as e:
                        handler_info["error"] = str(e)
                else:
                    handler_info = {
                        "type": handler_type,
                        "level": logging.getLevelName(handler.level),
                        "formatter": bool(handler.formatter),
                    }

                handlers_info.append(handler_info)

            logger_info = {
                "level": logging.getLevelName(logger.level),
                "propagate": logger.propagate,
                "handlers": handlers_info,
                "handler_count": len(logger.handlers),
            }

            loggers_info[name] = logger_info

            # Print info
            print(f"\nLogger: {name}")
            print(f"  Level: {logger_info['level']}")
            print(f"  Propagate: {logger_info['propagate']}")
            print(f"  Handlers: {logger_info['handler_count']}")

            for i, h_info in enumerate(handlers_info):
                print(f"    Handler {i + 1}: {h_info['type']}")
                if "file" in h_info:
                    print(f"      File: {h_info['file']}")
                    if "dir_writable" in h_info:
                        print(f"      Directory writable: {h_info['dir_writable']}")
                    if "file_writable" in h_info:
                        print(f"      File writable: {h_info['file_writable']}")
                print(f"      Level: {h_info['level']}")
                if "error" in h_info:
                    print(f"      ERROR: {h_info['error']}")
        else:
            print(f"\nLogger {name} not found in registered loggers")
            if name == "root":
                # Check root logger
                root = logging.getLogger()
                print(f"Root logger level: {logging.getLevelName(root.level)}")
                print(f"Root logger handlers: {len(root.handlers)}")

    return loggers_info
