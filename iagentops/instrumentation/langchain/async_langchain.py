import time
import importlib.util
import logging
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers
import json
import traceback
import inspect

WRAPPED_METHODS = [
    {"package": "langchain.llms.base", "object": "BaseLLM.apredict", "provider_attr": "client_name", "operation": "inference"},
    {"package": "langchain.llms.base", "object": "BaseLLM.agenerate", "provider_attr": "client_name", "operation": "inference"},

    # Async chains / workflows
    # {"package": "langchain.chains.base", "object": "BaseChain.arun", "provider_attr": None, "operation": "workflow"},

    {"package": "langchain_core.runnables.base", "object": "Runnable.ainvoke", "provider_attr": None, "operation": "workflow"},
    {"package": "langchain_core.runnables.base", "object": "Runnable.abatch", "provider_attr": None, "operation": "workflow"},

    # Async agents
    {"package": "langchain.agents.agent", "object": "Agent.arun", "provider_attr": None, "operation": "invoke_agent"},

    # Async tools and embeddings
    {"package": "langchain.tools.base", "object": "BaseTool.arun", "provider_attr": None, "operation": "tool"},
    {"package": "langchain.embeddings.base", "object": "Embeddings.aembed", "provider_attr": None, "operation": "embedding"},
]

from .callback_handler import IAgentOpsCallbackHandler

class AsyncLangChainInstrumentor:
    """Instrumentor for LangChain async LLM calls."""

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

        logger = logging.getLogger(__name__)
        for m in WRAPPED_METHODS:
            # Only wrap if the specific package path exists
            try:
                if importlib.util.find_spec(m["package"]) is None:
                    continue
            except Exception:
                # If package detection fails (e.g. module not found), skip it
                continue
            
            try:
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("operation"))
                )
            except Exception:
                logger.debug(f"Skipping wrap for {m['package']}:{m['object']}", exc_info=True)

    def _wrap(self, operation=None):
        def wrapper(wrapped, instance, args, kwargs):
            # Create handler
            handler = IAgentOpsCallbackHandler(
                tracer=self.tracer,
                agent_id=self.agent_id,
                service_name=self.service_name,
                environment=self.environment,
                system="langchain"
            )
            
            # Persist framework in context
            c_ctx = kwargs.copy()
            c_ctx["framework"] = "langchain"
            helpers.get_active_context(c_ctx)
            
            # Inject into callbacks safely using signature binding
            try:
                sig = inspect.signature(wrapped)
                # If instance is not None, it's a method call, sig likely includes 'self'
                if instance is not None:
                    bound = sig.bind(instance, *args, **kwargs)
                else:
                    bound = sig.bind(*args, **kwargs)
                
                # Check for callbacks in bound arguments
                callbacks = bound.arguments.get("callbacks", [])
                if callbacks is None:
                    callbacks = []
                if not isinstance(callbacks, list):
                    callbacks = [callbacks]
                
                # Avoid duplicate handlers
                if not any(isinstance(c, IAgentOpsCallbackHandler) for c in callbacks):
                    callbacks.append(handler)
                    bound.arguments["callbacks"] = callbacks
                    
                # Use bound args/kwargs to call the original function
                return wrapped(*bound.args, **bound.kwargs)
            except Exception:
                # If signature binding fails, try naive injection as fallback
                try:
                    if "callbacks" not in kwargs:
                        # Only add if not in kwargs; still risky if in args but better than nothing
                        kwargs["callbacks"] = [handler]
                    else:
                        cbs = kwargs["callbacks"]
                        if cbs is None: cbs = []
                        if not isinstance(cbs, list): cbs = [cbs]
                        if not any(isinstance(c, IAgentOpsCallbackHandler) for c in cbs):
                            cbs.append(handler)
                            kwargs["callbacks"] = cbs
                except:
                    pass
                return wrapped(*args, **kwargs)

        return wrapper

    def _detect_provider(self, instance, provider_attr):
        provider = getattr(instance, provider_attr, None)
        if provider and provider != "unknown":
            return provider.lower()
        return "unknown"
