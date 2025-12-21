import os
import threading
import atexit
import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

logger = logging.getLogger(__name__)

_initialized = False
_init_lock = threading.Lock()
_tracer_provider = None

import os
import socket
from urllib.parse import urlparse

def _get_local_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no traffic sent; used to discover local iface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return None

def _infer_server_host_port(collector_endpoint: str | None, exporter_protocol: str | None) -> tuple[str | None, int | None]:

    endpoint = (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or collector_endpoint
        or ""
    )

    host, port = None, None
    if endpoint:
        try:
            url = urlparse(endpoint)
            if not url.scheme and "://" not in endpoint:
                url = urlparse("http://" + endpoint)
            host = url.hostname
            port = url.port

            if port is None:
                if url.scheme in ("http", "https"):
                    port = 4318
                elif url.scheme in ("grpc", "otlp", "tcp"):
                    port = 4317
        except Exception:
            pass

    # Normalize localhost to LAN IP

    if host in (None, "localhost", "127.0.0.1"):
        local_ip = _get_local_ip()
        if local_ip:
            host = local_ip

    # Protocol defaults if still missing

    if exporter_protocol:
        p = exporter_protocol.lower()
        if "grpc" in p:
            host = host or (socket.getfqdn() or "localhost")
            port = port or 4317
            return host, int(port)

        if "http" in p:
            host = host or (socket.getfqdn() or "localhost")
            port = port or 4318
            return host, int(port)

    # Final fallback
    host = host or (socket.getfqdn() or "localhost")
    if host in ("localhost", "127.0.0.1"):
        local_ip = _get_local_ip()
        if local_ip:
            host = local_ip
    port = port or 4318

    return host, int(port)

def _shutdown_tracer():
    """Gracefully shutdown tracer provider to flush remaining spans."""
    global _tracer_provider
    if _tracer_provider:
        try:
            _tracer_provider.shutdown()
            logger.debug("iAgentOps tracer provider shutdown complete")
        except Exception as e:
            logger.warning(f"Error during tracer shutdown: {e}")

def setup_tracer(service_name: str | None = None, environment: str | None = None, sdk_version: str = "0.1.0", agent_id: str | None = None, server_address: str | None = None, server_port: int | None = None, collector_endpoint: str | None = None, use_console_exporter: bool = True, exporter_protocol: str = "http"):
    """Configure and return an OpenTelemetry tracer for iAgentOps.
    
    Args:
        service_name: Optional explicit service name. Falls back to OTEL_SERVICE_NAME or 'iagentops'.
        environment: Optional deployment environment. Falls back to IAGENTOPS_ENV / OTEL_ENVIRONMENT or 'development'.
        sdk_version: SDK version string (passed from iagentops.__version__)
        agent_id: Optional agent identifier for registry tracking.
        server_address: Optional server address for agent registry.
        server_port: Optional server port for agent registry.
        collector_endpoint: OTLP collector endpoint (e.g., 'http://localhost:4318' for HTTP or 'localhost:4317' for gRPC)
        use_console_exporter: Whether to also output to console (default: True)
        exporter_protocol: Protocol for OTLP exporter ('http' or 'grpc', default: 'http')
    """
    global _initialized, _tracer_provider

    service = service_name or os.getenv("OTEL_SERVICE_NAME") or "iagentops"
    env = environment or os.getenv("IAGENTOPS_ENV") or os.getenv("OTEL_ENVIRONMENT") or "development"

    if not _initialized:
        with _init_lock:
            if not _initialized:
                auto_host, auto_port = _infer_server_host_port(collector_endpoint, exporter_protocol)
                srv_addr = server_address or auto_host
                srv_port = server_port if server_port is not None else auto_port
                
                resource_attrs = {
                    "service.name": service,
                    "deployment.environment": env,
                    "telemetry.sdk.name": "iagentops",
                    "telemetry.sdk.language": "python",
                    "telemetry.sdk.version": sdk_version,
                }
                
                # Add agent registry attributes if provided
                if agent_id:
                    resource_attrs["agent.id"] = str(agent_id)
                if srv_addr:
                    resource_attrs["server.address"] = srv_addr
                if srv_port is not None:
                    try:
                        resource_attrs["server.port"] = int(srv_port)
                    except Exception:
                        resource_attrs["server.port"] = srv_port
                               
                resource = Resource.create(resource_attrs)
                _tracer_provider = TracerProvider(resource=resource)
                
                # Add exporters based on configuration
                exporters_added = False
                
                # Add OTLP exporter if collector endpoint is provided
                if collector_endpoint:
                    try:
                        if exporter_protocol == "grpc" or (not exporter_protocol and not collector_endpoint.startswith("http")):
                            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                            otlp_exporter = OTLPSpanExporter(endpoint=collector_endpoint)
                            logger.info(f"Using OTLP gRPC exporter: {collector_endpoint}")
                        else:
                            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                            otlp_exporter = OTLPSpanExporter(endpoint=collector_endpoint)
                            logger.info(f"Using OTLP HTTP exporter: {collector_endpoint}")
                        
                        _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                        exporters_added = True
                    except Exception as e:
                        logger.error(f"Failed to initialize OTLP exporter: {e}", exc_info=True)
                
                # Add console exporter if requested or as fallback
                if use_console_exporter or not exporters_added:
                    processor = BatchSpanProcessor(ConsoleSpanExporter())
                    _tracer_provider.add_span_processor(processor)
                    if use_console_exporter:
                        logger.info("Console exporter enabled")
                
                try:
                    trace.set_tracer_provider(_tracer_provider)
                except Exception as e:
                    logger.debug(f"TracerProvider already set or could not be set: {e}")
                
                # Initialize meter for agent registry metrics
                from iagentops.otel import metrics as metrics_module
                metrics_module._setup_meter(
                    resource=resource,
                    collector_endpoint=collector_endpoint,
                    use_console_exporter=use_console_exporter,
                    exporter_protocol=exporter_protocol
                )
                
                # Register graceful shutdown
                atexit.register(_shutdown_tracer)
                
                _initialized = True
                logger.info(f"iAgentOps tracer initialized: service={service}, env={env}, version={sdk_version}, agent={agent_id}")
                
                try:
                    from opentelemetry.instrumentation.logging import LoggingInstrumentor
                    LoggingInstrumentor().instrument(set_logging_format=True)
                    logger.info("Logging instrumentation enabled for trace correlation")
                except Exception as e:
                    logger.debug(f"Logging instrumentation not available: {e}")

    return trace.get_tracer(service)

class TraceManager:
    """
    A manager for creating and managing traces and spans.
    """

    def __init__(self):
        self.tracer = trace.get_tracer_provider().get_tracer("iagentops")

    def start_trace(self, name: str):
        """
        Start a new trace.

        Args:
            name: The name of the trace.

        Returns:
            A context manager for the trace span.
        """
        return self.tracer.start_as_current_span(name)

    def add_span_attributes(self, span, attributes: dict):
        """
        Add attributes to a span.

        Args:
            span: The span to which attributes will be added.
            attributes: A dictionary of attributes to add.
        """
        for key, value in attributes.items():
            span.set_attribute(key, value)

    def end_trace(self, span):
        """
        End the given trace span.

        Args:
            span: The span to end.
        """
        span.end()

def create_trace(trace_name: str, attributes: dict = None):
    """
    Create and manage a trace with the given name and optional attributes.

    Args:
        trace_name: The name of the trace.
        attributes: Optional dictionary of attributes to add to the trace.

    Returns:
        The created trace span.
    """
    trace_manager = TraceManager()
    # Merge default GenAI attributes with provided attributes
    default_attributes = GenAIAttributes.get_attributes()
    if attributes:
        default_attributes.update(attributes)

    # Add framework detection explicitly
    from iagentops._instrumentors import detect_frameworks
    frameworks = detect_frameworks()
    if "langgraph" in frameworks:
        default_attributes["agent.framework"] = "langgraph"
    elif "langchain" in frameworks:
        default_attributes["agent.framework"] = "langchain"

    with trace_manager.start_trace(trace_name) as span:
        trace_manager.add_span_attributes(span, default_attributes)
        return span

class TraceReporter:
    """
    A reporter for collecting and exporting trace data.
    """

    def __init__(self):
        self.exporter = ConsoleSpanExporter()

    def report_trace(self, span):
        """
        Export the given trace span for reporting.

        Args:
            span: The span to export.
        """
        self.exporter.export([span])

class GenAIAttributes:
    """
    A class to define and manage attributes for GenAI traces and spans.
    """

    @staticmethod
    def get_attributes():
        """
        Returns a dictionary of GenAI attributes.

        Returns:
            dict: A dictionary containing GenAI attributes and their descriptions.
        """
        return {
            "gen_ai.conversation.id": "Unique conversation/session identifier used to correlate messages in a conversation.",
            "gen_ai.data_source.id": "Data source identifier for retrieval or knowledge sources.",
            "gen_ai.input.messages": "Chat history provided to the model as input (structured).",
            "gen_ai.operation.name": "Name of the operation being performed (e.g., chat, generate_content, etc.).",
            "gen_ai.output.messages": "Messages returned by the model (choices/candidates).",
            "gen_ai.output.type": "Requested output modality (e.g., text, json, image).",
            "gen_ai.provider.name": "Generative AI provider as identified by instrumentation (e.g., openai, aws.bedrock, etc.).",
            "gen_ai.request.choice.count": "Target number of candidate completions to return.",
            "gen_ai.request.encoding_formats": "Embedding encoding formats requested (e.g., base64, float, binary).",
            "gen_ai.request.frequency_penalty": "Frequency penalty setting.",
            "gen_ai.request.model": "Name of the GenAI model requested.",
            "gen_ai.request.presence_penalty": "Presence penalty setting.",
            "gen_ai.request.seed": "Seed value to increase reproducibility.",
            "gen_ai.request.stop_sequences": "List of sequences that will stop further token generation.",
            "gen_ai.request.temperature": "Temperature setting for sampling.",
            "gen_ai.request.top_p": "Top-p (nucleus) sampling setting.",
            "gen_ai.response.model": "Name of the model that generated the response.",
            "gen_ai.system_instructions": "System message or instructions provided separately from chat history.",
            "gen_ai.tool.call.id": "Tool call identifier.",
            "gen_ai.tool.description": "Tool description.",
            "gen_ai.tool.name": "Name of the tool utilized by the agent.",
            "gen_ai.tool.type": "Type of the tool utilized by the agent (function | extension | datastore).",
            "gen_ai.usage.input_tokens": "Number of tokens used in the GenAI input (prompt).",
            "gen_ai.usage.output_tokens": "Number of tokens used in the GenAI output (completion).",
            "server.address": "GenAI Server Address.",
            "server.port": "GenAI Server Port."
        }


class GenAISpans:
    """
    A class to define and manage spans for GenAI operations.
    """

    @staticmethod
    def create_inference_span(tracer, name="Inference span", attributes=None):
        """
        Create an inference span.

        Args:
            tracer: The tracer instance to create the span.
            name: The name of the span.
            attributes: Optional dictionary of attributes to add to the span.

        Returns:
            The created span.
        """
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span

    @staticmethod
    def create_invoke_agent_span(tracer, agent_name, attributes=None):
        """
        Create an invoke agent span.

        Args:
            tracer: The tracer instance to create the span.
            agent_name: The name of the agent being invoked.
            attributes: Optional dictionary of attributes to add to the span.

        Returns:
            The created span.
        """
        span_name = f"invoke_agent ({agent_name})"
        with tracer.start_as_current_span(span_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span

    @staticmethod
    def create_create_agent_span(tracer, agent_name, attributes=None):
        """
        Create a create agent span.

        Args:
            tracer: The tracer instance to create the span.
            agent_name: The name of the agent being created.
            attributes: Optional dictionary of attributes to add to the span.

        Returns:
            The created span.
        """
        span_name = f"create_agent ({agent_name})"
        with tracer.start_as_current_span(span_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span

    @staticmethod
    def create_embedding_span(tracer, name="embedding span", attributes=None):
        """
        Create an embedding span.

        Args:
            tracer: The tracer instance to create the span.
            name: The name of the span.
            attributes: Optional dictionary of attributes to add to the span.

        Returns:
            The created span.
        """
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span
