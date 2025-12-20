"""iAgentOps - Auto-instrumentation for AI Agents and LLMs"""

from iagentops._instrumentors import auto_instrument
from iagentops.otel.metrics import (
    get_agent_metrics,
    emit_agent_metrics,
)
from iagentops.otel.metrics import set_agent_id

__version__ = "0.1.0"

def init(
    api_key=None,
    environment="development",
    disabled=False,
    agent_id=None,
    server_address=None,
    server_port=None,
    service_name="iagentops",
    collector_endpoint=None,
    use_console_exporter=True,
    exporter_protocol=None,
    **kwargs
):
    """
    Initialize iAgentOps auto-instrumentation.
    
    Args:
        api_key: Optional API key for remote telemetry (not used for console output)
        environment: Environment name (development/staging/production)
        disabled: Set to True to disable instrumentation
        agent_id: Optional unique agent identifier for registry
        server_address: Optional server address
        server_port: Optional server port
        service_name: Service name for traces (default: iagentops)
        collector_endpoint: OTLP collector endpoint (e.g., 'http://localhost:4318' for HTTP or 'localhost:4317' for gRPC)
        use_console_exporter: Whether to also output to console (default: True for development)
        exporter_protocol: Protocol for OTLP exporter ('http' or 'grpc', default: 'http')
        **kwargs: Additional configuration options
    
    Example:
        import iagentops
        
        # Send to OTLP collector only
        iagentops.init(
            environment="production",
            agent_id="my-agent-1",
            collector_endpoint="http://localhost:4318",
            use_console_exporter=False
        )
        
        # Development mode with both console and collector
        iagentops.init(
            environment="development",
            agent_id="my-agent-1",
            collector_endpoint="http://localhost:4318",
            use_console_exporter=True
        )
    """
    if disabled:
        return
    
    from opentelemetry.sdk.resources import Resource
    from iagentops.otel.metrics import _setup_meter

    _setup_meter(
        resource=Resource.create({"service.name": service_name}),
        collector_endpoint=collector_endpoint,
        use_console_exporter=use_console_exporter,
        exporter_protocol=exporter_protocol
    )
    
    # Basic input validation
    if service_name is not None and not isinstance(service_name, str):
        raise TypeError(f"service_name must be a string, got {type(service_name)}")
    if isinstance(service_name, str) and not service_name.strip():
        raise ValueError("service_name cannot be empty")
    
    if agent_id is not None and not isinstance(agent_id, str):
        raise TypeError(f"agent_id must be a string, got {type(agent_id)}")
    
    set_agent_id(agent_id)
    
    # Auto-detect and instrument available frameworks
    auto_instrument(
        api_key=api_key, 
        environment=environment, 
        service_name=service_name, 
        sdk_version=__version__,
        agent_id=agent_id,
        server_address=server_address,
        server_port=server_port,
        collector_endpoint=collector_endpoint,
        use_console_exporter=use_console_exporter,
        exporter_protocol=exporter_protocol,
        **kwargs
    )


from iagentops.context import set_context

__all__ = [
    "init",
    "set_context",
    "__version__",
    "get_agent_metrics",
    "emit_agent_metrics",
]