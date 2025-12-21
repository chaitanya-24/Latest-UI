"""Auto-instrumentation registry and detection"""

import importlib.util
import logging

from iagentops.instrumentation.langchain.langchain import LangChainInstrumentor
from iagentops.instrumentation.langchain.async_langchain import AsyncLangChainInstrumentor
from iagentops.instrumentation.langgraph.langgraph import LangGraphInstrumentor
from iagentops.instrumentation.langgraph.async_langgraph import AsyncLangGraphInstrumentor
from iagentops.instrumentation.crewai.crewai import CrewAIInstrumentor
from iagentops.instrumentation.crewai.async_crewai import AsyncCrewAIInstrumentor
from iagentops.instrumentation.adk.adk import ADKInstrumentor
from iagentops.instrumentation.adk.async_adk import AsyncADKInstrumentor
from iagentops.instrumentation.mcp.mcp import MCPInstrumentor
from iagentops.instrumentation.vectorstore.vectorstore import VectorStoreInstrumentor
from iagentops.instrumentation.llm_providers.openai import OpenAIInstrumentor
from iagentops.instrumentation.llm_providers.anthropic import AnthropicInstrumentor
from iagentops.otel import metrics

logger = logging.getLogger(__name__)

# Registry of all available instrumentors
INSTRUMENTORS = {
    "langchain": {
        "sync": LangChainInstrumentor,
        "async": AsyncLangChainInstrumentor,
        "package": "langchain",
    },
    "langgraph": {
        "sync": LangGraphInstrumentor,
        "async": AsyncLangGraphInstrumentor,
        "package": "langgraph",
    },
    "crewai": {
        "sync": CrewAIInstrumentor,
        "async": AsyncCrewAIInstrumentor,
        "package": "crewai",
    },
    "adk": {
        "sync": ADKInstrumentor,
        "async": AsyncADKInstrumentor,
        "package": "google.adk",  
    },
    "mcp": {
        "sync": MCPInstrumentor,
        "async": MCPInstrumentor,
        "package": "mcp",
    },
    "chromadb": {
        "sync": VectorStoreInstrumentor,
        "async": VectorStoreInstrumentor,
        "package": "chromadb",
    },
    "pinecone": {
        "sync": VectorStoreInstrumentor,
        "async": VectorStoreInstrumentor,
        "package": "pinecone",
    },
    "qdrant_client": {
        "sync": VectorStoreInstrumentor,
        "async": VectorStoreInstrumentor,
        "package": "qdrant_client",
    },
    "faiss": {
        "sync": VectorStoreInstrumentor,
        "async": VectorStoreInstrumentor,
        "package": "faiss",
    },
    "openai": {
        "sync": OpenAIInstrumentor,
        "async": OpenAIInstrumentor,
        "package": "openai",
    },
    "anthropic": {
        "sync": AnthropicInstrumentor,
        "async": AnthropicInstrumentor,
        "package": "anthropic",
    },
}

_instrumented = set()

def is_package_available(package_name):
    """Check if a package is installed."""
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False

def detect_frameworks():
    """Auto-detect which AI frameworks are installed."""
    detected = []
    for name, config in INSTRUMENTORS.items():
        if is_package_available(config["package"]):
            detected.append(name)
            logger.info(f"Detected framework: {name}")
    return detected

def auto_instrument(api_key=None, environment="development", service_name="iagentops", sdk_version="0.1.0", agent_id=None, server_address=None, server_port=None, collector_endpoint=None, use_console_exporter=True, exporter_protocol="http", **kwargs):
    """
    Automatically detect and instrument all available AI frameworks.
    
    Args:
        api_key: Optional API key for remote telemetry
        environment: Environment name
        service_name: Service name for traces (default: iagentops)
        sdk_version: SDK version string
        agent_id: Optional agent identifier for registry tracking
        server_address: Optional server address for agent registry
        server_port: Optional server port for agent registry
        collector_endpoint: OTLP collector endpoint
        use_console_exporter: Whether to also output to console
        exporter_protocol: Protocol for OTLP exporter ('http' or 'grpc')
        **kwargs: Additional options
    """
    from iagentops.otel import metrics
    
    # Register agent on first instrumentation
    if agent_id:
        metrics.increment_registrations()
        logger.info(f"Agent registered: {agent_id}")
    
    # Auto-detect frameworks
    frameworks = detect_frameworks()
    
    if not frameworks:
        logger.warning(
            "No supported AI frameworks detected. "
            "Install langchain or langgraph to enable instrumentation."
        )
        return
    
    logger.info(f"Auto-instrumenting frameworks: {', '.join(frameworks)}")
    
    # Instrument each detected framework
    for framework in frameworks:
        if framework in _instrumented:
            logger.debug(f"Framework {framework} already instrumented, skipping")
            continue
        
        config = INSTRUMENTORS[framework]
        
        # Instrument both sync and async versions
        for variant in ["sync", "async"]:
            try:
                instr_class = config[variant]
                instr = instr_class()
                instr.instrument(
                    service_name=service_name, 
                    environment=environment, 
                    sdk_version=sdk_version,
                    agent_id=agent_id,
                    server_address=server_address,
                    server_port=server_port,
                    collector_endpoint=collector_endpoint,
                    use_console_exporter=use_console_exporter,
                    exporter_protocol=exporter_protocol,
                    **kwargs
                )
                logger.info(f"Instrumented {framework} ({variant})")
            except Exception as e:
                logger.error(f"Failed to instrument {framework} ({variant}): {e}", exc_info=True)
        
        _instrumented.add(framework)
    
    logger.info("iAgentOps auto-instrumentation complete")
