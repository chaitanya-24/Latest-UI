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
import inspect
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
            
            try:
                sig = inspect.signature(wrapped)
                if instance is not None:
                    bound = sig.bind(instance, *args, **kwargs)
                else:
                    bound = sig.bind(*args, **kwargs)
                
                # Search for where to put callbacks. 
                # LangChain elements can have them in:
                # 1. 'callbacks' named parameter
                # 2. 'config' named parameter (as config['callbacks'])
                # 3. 'kwargs' parameter (as kwargs['callbacks'] or kwargs['config']['callbacks'])
                
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
                            
                # Try inside VAR_KEYWORD (**kwargs)
                if not found_in_config:
                    kwargs_param = next((p for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD), None)
                    if kwargs_param and kwargs_param.name in bound.arguments:
                        extra_kwargs = bound.arguments[kwargs_param.name]
                        
                        # Check extra_kwargs['callbacks']
                        if "callbacks" in extra_kwargs:
                            cbs = extra_kwargs["callbacks"]
                            if cbs is None: cbs = []
                            if not isinstance(cbs, list): cbs = [cbs]
                            else: cbs = list(cbs)
                            if not any(isinstance(c, IAgentOpsCallbackHandler) for c in cbs):
                                cbs.append(handler)
                                extra_kwargs["callbacks"] = cbs
                                found_in_config = True
                        
                        # Check extra_kwargs['config']['callbacks']
                        if not found_in_config and "config" in extra_kwargs and isinstance(extra_kwargs["config"], dict):
                            config = extra_kwargs["config"]
                            cbs = config.get("callbacks", [])
                            if cbs is None: cbs = []
                            if not isinstance(cbs, list): cbs = [cbs]
                            else: cbs = list(cbs)
                            if not any(isinstance(c, IAgentOpsCallbackHandler) for c in cbs):
                                cbs.append(handler)
                                config["callbacks"] = cbs
                                found_in_config = True

                # If we still haven't found a place to put it and the method likely accepts it,
                # we could add it to kwargs, but that's what causes "multiple values" if we are wrong.
                # So we ONLY add to kwargs if 'callbacks' is NOT in 'sig.parameters' at all
                # AND we didn't find it in config.
                if not found_in_config and "callbacks" not in sig.parameters:
                    kwargs_param = next((p for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD), None)
                    if kwargs_param:
                        if kwargs_param.name not in bound.arguments:
                            bound.arguments[kwargs_param.name] = {}
                        if "callbacks" not in bound.arguments[kwargs_param.name]:
                            bound.arguments[kwargs_param.name]["callbacks"] = [handler]

                return wrapped(*bound.args, **bound.kwargs)
                
            except Exception:
                # Fallback to naive but only if absolutely necessary
                return wrapped(*args, **kwargs)

        return wrapper

    def _detect_provider(self, instance, provider_attr):
        # Kept for backward compatibility if needed, though CallbackHandler handles this now
        provider = getattr(instance, provider_attr, None)
        if provider and provider != "unknown":
            return provider.lower()
        return "unknown"
