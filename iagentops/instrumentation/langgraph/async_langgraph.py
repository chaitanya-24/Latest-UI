import time
import importlib.util
import logging
import inspect
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers
from iagentops.instrumentation.langchain.callback_handler import IAgentOpsCallbackHandler

WRAPPED_METHODS = [
    # LangGraph CompiledGraph (async)
    {"package": "langgraph.graph.graph", "object": "CompiledGraph.ainvoke", "operation": "workflow"},
    {"package": "langgraph.graph.graph", "object": "CompiledGraph.astream", "operation": "workflow"},
    {"package": "langgraph.graph.graph", "object": "CompiledStateGraph.ainvoke", "operation": "workflow"},
    {"package": "langgraph.graph.graph", "object": "CompiledStateGraph.astream", "operation": "workflow"},
]

class AsyncLangGraphInstrumentor:
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
            if importlib.util.find_spec("langgraph") is None:
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
                
                # Check for config in bound arguments
                config = bound.arguments.get("config", {})
                if config is None:
                    config = {}
                
                # In config, we have 'callbacks'
                callbacks = config.get("callbacks", [])
                if callbacks is None:
                    callbacks = []
                if not isinstance(callbacks, list):
                    callbacks = [callbacks]
                
                # Avoid duplicate handlers
                if not any(isinstance(c, IAgentOpsCallbackHandler) for c in callbacks):
                    callbacks.append(handler)
                    config["callbacks"] = callbacks
                    bound.arguments["config"] = config
                
                return wrapped(*bound.args, **bound.kwargs)
            except Exception:
                # Fallback naive injection
                try:
                    config = kwargs.get("config", {})
                    if config is None: config = {}
                    
                    if not isinstance(config, dict):
                        return wrapped(*args, **kwargs)
                        
                    callbacks = config.get("callbacks", [])
                    if callbacks is None: callbacks = []
                    if not isinstance(callbacks, list): callbacks = [callbacks]
                    
                    if not any(isinstance(c, IAgentOpsCallbackHandler) for c in callbacks):
                        callbacks.append(handler)
                        config["callbacks"] = callbacks
                        kwargs["config"] = config
                except:
                    pass
                return wrapped(*args, **kwargs)

        return wrapper
