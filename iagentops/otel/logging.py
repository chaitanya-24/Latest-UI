import logging
import os
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk.resources import Resource


# -----------------------------------------------------------------------------
# INTERNAL STATE (prevents double initialization)
# -----------------------------------------------------------------------------
_INITIALIZED = False


def setup_logging(service_name: str, otlp_endpoint: str, log_level=logging.INFO):
    """
    Set up OpenTelemetry logging for Loki via OTLP exporter.

    IMPORTANT:
    - Safe to call multiple times
    - Does NOT override global logging config
    - Prevents duplicate log records
    """

    global _INITIALIZED
    if _INITIALIZED:
        return logging.getLogger(service_name)

    # -------------------------------------------------------------------------
    # Resource (used by Loki labels via OTel Collector)
    # -------------------------------------------------------------------------
    resource = Resource(
        attributes={
            "service.name": service_name,
        }
    )

    # -------------------------------------------------------------------------
    # OTLP Log Exporter
    # -------------------------------------------------------------------------
    otlp_log_exporter = OTLPLogExporter(
        endpoint=otlp_endpoint,
        insecure=True,
    )

    # -------------------------------------------------------------------------
    # LoggerProvider (GLOBAL, MUST BE SINGLETON)
    # -------------------------------------------------------------------------
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(otlp_log_exporter)
    )

    set_logger_provider(logger_provider)

    # -------------------------------------------------------------------------
    # LoggingHandler (attach ONCE to root logger)
    # -------------------------------------------------------------------------
    otel_handler = LoggingHandler(
        level=log_level,
        logger_provider=logger_provider,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers
    if not any(isinstance(h, LoggingHandler) for h in root_logger.handlers):
        root_logger.addHandler(otel_handler)

    # -------------------------------------------------------------------------
    # Service Logger
    # -------------------------------------------------------------------------
    logger = logging.getLogger(service_name)
    logger.propagate = True

    logger.info(
        "OTEL logging initialized for %s (endpoint=%s)",
        service_name,
        otlp_endpoint,
    )

    _INITIALIZED = True
    return logger


def _auto_setup_logging():
    """
    Automatically initialize logging using environment variables.

    Environment variables:
      IAGENTOPS_SERVICE_NAME   (default: iagentops-app)
      IAGENTOPS_OTLP_ENDPOINT (default: localhost:4317)
      IAGENTOPS_LOG_LEVEL     (default: INFO)
    """

    service_name = os.getenv("IAGENTOPS_SERVICE_NAME", "iagentops-app")
    otlp_endpoint = os.getenv("IAGENTOPS_OTLP_ENDPOINT", "localhost:4317")

    log_level_str = os.getenv("IAGENTOPS_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    setup_logging(service_name, otlp_endpoint, log_level)


def get_sdk_logger():
    """
    Logger for SDK internal logs.
    """
    return logging.getLogger("iagentops-sdk")


# -----------------------------------------------------------------------------
# Auto-initialize on import (safe, idempotent)
# -----------------------------------------------------------------------------
_auto_setup_logging() 