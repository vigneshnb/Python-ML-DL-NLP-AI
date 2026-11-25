"""Central logging setup."""
from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging(level: str = "INFO") -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
            "loggers": {
                # Keep third-party noise down
                "tensorflow": {"level": "ERROR"},
                "absl": {"level": "ERROR"},
                "PIL": {"level": "WARNING"},
                "h5py": {"level": "WARNING"},
                "werkzeug": {"level": "INFO"},
            },
        }
    )
    logging.getLogger(__name__).debug("Logging configured at %s", level)
