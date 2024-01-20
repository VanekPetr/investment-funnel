import logging
import os
import sys
from pprint import pformat
from typing import List

from loguru import logger

LOG_LEVEL = logging.getLevelName(os.environ.get("LOG_LEVEL", "DEBUG"))
JSON_LOGS = True if os.environ.get("JSON_LOGS", "0") == "1" else False


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def format_record(record: dict) -> str:
    """
    Custom format for loguru loggers.
    Uses pformat for log any data like request/response body during debug.
    Works with logging if loguru handler it.
    """
    format_string = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        + "<level>{level: <8}</level> | "
        + "<level>{message}</level>"
    )

    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(
            record["extra"]["payload"], indent=4, compact=True, width=88
        )
        format_string += " | <level>{extra[payload]}</level>"

    format_string += "{exception}\n"
    return format_string


def setup_logging(
    log_names: List[str] = [
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.asgi",
        "fastapi",
        "gunicorn.access",
        "gunicorn.error",
    ]
):
    # intercept everything at the root logger

    # remove every other logger's handlers
    # and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = False

    # configure loguru
    if log_names:
        for name in log_names:
            logging.getLogger(name).handlers = [InterceptHandler()]
            logging.getLogger(name).setLevel(LOG_LEVEL)
    else:
        logging.root.handlers = [InterceptHandler()]
        logging.root.setLevel(LOG_LEVEL)
    handlers = []

    sysout_handler = {
        "sink": sys.stdout,
        "level": LOG_LEVEL,
        "format": format_record,
        "diagnose": False,
    }
    handlers.append(sysout_handler)

    logger.remove()
    logger.configure(handlers=handlers)
