import time
import importlib.util
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers
import json
import traceback
try:
    from langchain_core.callbacks import BaseCallbackHandler as _LCBaseCallbackHandler
except Exception:
    _LCBaseCallbackHandler = object

WRAPPED_METHODS = [
    # LLM synchronous inference
    {"package": "langchain.llms.base", "object": "BaseLLM.predict", "provider_attr": "client_name", "operation": "inference"},
    {"package": "langchain.llms.base", "object": "BaseLLM.generate", "provider_attr": "client_name", "operation": "inference"},
    {"package": "langchain_core.language_models.chat_models", "object": "BaseChatModel.invoke", "provider_attr": "client_name", "operation": "inference"},
    {"package": "langchain_core.language_models.chat_models", "object": "BaseChatModel.generate", "provider_attr": "client_name", "operation": "inference"},

    # Chains / workflows
    {"package": "langchain.chains.base", "object": "BaseChain.run", "provider_attr": None, "operation": "workflow"},

    # Agents and agent executors
    {"package": "langchain.agents.agent", "object": "Agent.run", "provider_attr": None, "operation": "invoke_agent"},
    {"package": "langchain.agents.agent", "object": "AgentExecutor.run", "provider_attr": None, "operation": "invoke_agent"},

    # Tools
    {"package": "langchain.tools.base", "object": "BaseTool.run", "provider_attr": None, "operation": "tool"},

    # Embeddings
    {"package": "langchain.embeddings.base", "object": "Embeddings.embed", "provider_attr": None, "operation": "embedding"},
    {"package": "langchain.embeddings.base", "object": "Embeddings.embed_documents", "provider_attr": None, "operation": "embedding"},
]

from .callback_handler import IAgentOpsCallbackHandler

class LangChainInstrumentor:
    def instrument(self, service_name="iagentops", environment="development", sdk_version="0.1.0", agent_id=None, server_address=None, server_port=None, collector_endpoint=None, use_console_exporter=True, exporter_protocol="http", **kwargs):
        self.tracer = tracing.setup_tracer(
            service_name=service_name, 
            environment=environment, 
            sdk_version=sdk_version,
            agent_id=agent_id,
            server_address=server_address,
            server_port=server_port,
            collector_endpoint=collector_endpoint,
            use_console_exporter=use_console_exporter,
            exporter_protocol=exporter_protocol
        )
        self.service_name = service_name
        self.environment = environment
        self.agent_id = agent_id
        
        try:
            if importlib.util.find_spec("langchain") is None:
                return
        except Exception:
            return
        
        for m in WRAPPED_METHODS:
            try:
                if importlib.util.find_spec(m["package"]) is None:
                    continue
            except Exception:
                continue

            try:
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("operation"))
                )
            except Exception:
                continue

    def _wrap(self, operation=None):
        def wrapper(wrapped, instance, args, kwargs):
            # Create our callback handler
            handler = IAgentOpsCallbackHandler(
                tracer=self.tracer,
                agent_id=self.agent_id,
                service_name=self.service_name,
                environment=self.environment
            )
            
            # Inject into callbacks
            try:
                callbacks = kwargs.get("callbacks", [])
                if callbacks is None:
                    callbacks = []
                if not isinstance(callbacks, list):
                    callbacks = [callbacks]
                
                # Avoid duplicate handlers (e.g. if invoke calls generate and both are wrapped)
                if not any(isinstance(c, IAgentOpsCallbackHandler) for c in callbacks):
                    callbacks.append(handler)
                    kwargs["callbacks"] = callbacks
            except Exception:
                # If for some reason we can't inject callbacks (key error, immutable, etc), we might lose telemetry
                pass

            return wrapped(*args, **kwargs)

        return wrapper

    def _detect_provider(self, instance, provider_attr):
        # Kept for backward compatibility if needed, though CallbackHandler handles this now
        provider = getattr(instance, provider_attr, None)
        if provider and provider != "unknown":
            return provider.lower()
        return "unknown"
