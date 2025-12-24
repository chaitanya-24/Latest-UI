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
from .callback_handler import IAgentOpsCallbackHandler

WRAPPED_METHODS = [
    # LLM synchronous inference
    {"package": "langchain.llms.base", "object": "BaseLLM.predict", "operation": "inference"},
    {"package": "langchain.llms.base", "object": "BaseLLM.generate", "operation": "inference"},
    {"package": "langchain_core.language_models.chat_models", "object": "BaseChatModel.invoke", "operation": "inference"},
    {"package": "langchain_core.language_models.chat_models", "object": "BaseChatModel.generate", "operation": "inference"},
    
    # Generic Runnables (highly inclusive)
    {"package": "langchain_core.runnables.base", "object": "Runnable.invoke", "operation": "workflow"},
    
    # Chains / workflows
    {"package": "langchain.chains.base", "object": "BaseChain.run", "operation": "workflow"},

    # Agents and agent executors
    {"package": "langchain.agents.agent", "object": "Agent.run", "operation": "invoke_agent"},
    {"package": "langchain.agents.agent", "object": "AgentExecutor.run", "operation": "invoke_agent"},

    # Tools
    {"package": "langchain.tools.base", "object": "BaseTool.run", "operation": "tool"},

    # Embeddings
    {"package": "langchain.embeddings.base", "object": "Embeddings.embed", "operation": "embedding"},
    {"package": "langchain.embeddings.base", "object": "Embeddings.embed_documents", "operation": "embedding"},
]

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
            if importlib.util.find_spec("langchain") is None and importlib.util.find_spec("langchain_core") is None:
                return
        except Exception:
            return
        
        for m in WRAPPED_METHODS:
            try:
                if importlib.util.find_spec(m["package"]) is None:
                    continue
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("operation"))
                )
            except Exception:
                continue

    def _wrap(self, operation=None):
        def wrapper(wrapped, instance, args, kwargs):
            # Re-entrancy guard
            if helpers._IN_INSTRUMENTATION.get():
                return wrapped(*args, **kwargs)
                
            token = helpers._IN_INSTRUMENTATION.set(True)
            try:
                # Detect framework
                framework = helpers.detect_agent_framework(instance)
                
                # Create handler
                handler = IAgentOpsCallbackHandler(
                    tracer=self.tracer,
                    agent_id=self.agent_id,
                    service_name=self.service_name,
                    environment=self.environment,
                    system=framework
                )
                
                # Persist in context
                c_ctx = kwargs.copy()
                c_ctx["framework"] = framework
                helpers.get_active_context(c_ctx)
                
                try:
                    sig = inspect.signature(wrapped)
                    try:
                        bound = sig.bind(instance, *args, **kwargs)
                    except TypeError:
                        bound = sig.bind(*args, **kwargs)
                    
                    found_in_config = False
                    
                    if "callbacks" in bound.arguments:
                        cbs = bound.arguments["callbacks"]
                        if cbs is None: cbs = []
                        if not isinstance(cbs, list): cbs = [cbs]
                        else: cbs = list(cbs)
                        if not any(isinstance(c, IAgentOpsCallbackHandler) for c in cbs):
                            cbs.append(handler)
                            bound.arguments["callbacks"] = cbs
                            found_in_config = True
                            
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
                    
                    if not found_in_config and "callbacks" not in sig.parameters:
                        kwargs_param = next((p for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD), None)
                        if kwargs_param:
                            if kwargs_param.name not in bound.arguments:
                                bound.arguments[kwargs_param.name] = {}
                            if "callbacks" not in bound.arguments[kwargs_param.name]:
                                bound.arguments[kwargs_param.name]["callbacks"] = [handler]

                    return wrapped(*bound.args, **bound.kwargs)
                except Exception:
                    if "callbacks" not in kwargs:
                        kwargs["callbacks"] = [handler]
                    elif isinstance(kwargs["callbacks"], list):
                        if not any(isinstance(c, IAgentOpsCallbackHandler) for c in kwargs["callbacks"]):
                            kwargs["callbacks"].append(handler)
                    return wrapped(*args, **kwargs)
            finally:
                helpers._IN_INSTRUMENTATION.reset(token)

        return wrapper
