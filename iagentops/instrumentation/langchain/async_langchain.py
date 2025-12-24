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
            # Re-entrancy guard
            if helpers._IN_INSTRUMENTATION.get():
                return wrapped(*args, **kwargs)
            
            token = helpers._IN_INSTRUMENTATION.set(True)
            try:
                # Detect actual framework (might be langgraph even if called via langchain)
                framework = helpers.detect_agent_framework(instance)
                
                # Create handler
                handler = IAgentOpsCallbackHandler(
                    tracer=self.tracer,
                    agent_id=self.agent_id,
                    service_name=self.service_name,
                    environment=self.environment,
                    system=framework
                )
                
                # Persist framework in context
                c_ctx = kwargs.copy()
                c_ctx["framework"] = framework
                helpers.get_active_context(c_ctx)
                
                # Inject into callbacks safely
                try:
                    sig = inspect.signature(wrapped)
                    try:
                        bound = sig.bind(instance, *args, **kwargs)
                    except TypeError:
                        # Fallback for already-bound or wrapper methods
                        bound = sig.bind(*args, **kwargs)
                    
                    found_in_config = False
                    
                    # Try 'callbacks' top-level parameter
                    if "callbacks" in bound.arguments:
                        cbs = bound.arguments["callbacks"]
                        if cbs is None: cbs = []
                        if not isinstance(cbs, list): cbs = [cbs]
                        else: cbs = list(cbs)
                        
                        if not any(isinstance(c, IAgentOpsCallbackHandler) for c in cbs):
                            cbs.append(handler)
                            bound.arguments["callbacks"] = cbs
                            found_in_config = True
                    
                    # Try 'config' top-level parameter
                    if not found_in_config and "config" in bound.arguments:
                        config = bound.arguments["config"]
                        if isinstance(config, dict):
                            cbs = config.get("callbacks", [])
                            if cbs is None: cbs = []
                            if not isinstance(cbs, list): cbs = [cbs]
                            else: cbs = list(cbs)
                            
                            if not any(isinstance(c, IAgentOpsCallbackHandler) for c in cbs):
                                cbs.append(handler)
                                config["callbacks"] = cbs
                                found_in_config = True
                    
                    if not found_in_config:
                        # Fallback for Runnable.ainvoke etc.
                        if "callbacks" not in sig.parameters:
                            kwargs_param = next((p for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD), None)
                            if kwargs_param:
                                if kwargs_param.name not in bound.arguments:
                                    bound.arguments[kwargs_param.name] = {}
                                if "callbacks" not in bound.arguments[kwargs_param.name]:
                                    bound.arguments[kwargs_param.name]["callbacks"] = [handler]

                    return wrapped(*bound.args, **bound.kwargs)
                except Exception:
                    # Naive fallback
                    if "callbacks" not in kwargs:
                        kwargs["callbacks"] = [handler]
                    elif isinstance(kwargs["callbacks"], list):
                        if not any(isinstance(c, IAgentOpsCallbackHandler) for c in kwargs["callbacks"]):
                            kwargs["callbacks"].append(handler)
                    return wrapped(*args, **kwargs)
            finally:
                helpers._IN_INSTRUMENTATION.reset(token)

        return wrapper

    def _detect_provider(self, instance, provider_attr):
        provider = getattr(instance, provider_attr, None)
        if provider and provider != "unknown":
            return provider.lower()
        return "unknown"
