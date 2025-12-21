import time
import importlib.util
import logging
import inspect
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers
from iagentops.instrumentation.langchain.callback_handler import IAgentOpsCallbackHandler

WRAPPED_METHODS = [
    # LangGraph CompiledGraph (sync)
    {"package": "langgraph.graph.graph", "object": "CompiledGraph.invoke", "operation": "workflow"},
    {"package": "langgraph.graph.graph", "object": "CompiledGraph.stream", "operation": "workflow"},
    {"package": "langgraph.graph.graph", "object": "CompiledStateGraph.invoke", "operation": "workflow"},
    {"package": "langgraph.graph.graph", "object": "CompiledStateGraph.stream", "operation": "workflow"},
]

class LangGraphInstrumentor:
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
            
            # LangGraph Config logic: 
            # In LangGraph, callbacks are usually passed inside 'config' dict: 
            # invoke(inputs, config={"callbacks": [...]})
            
            # Get or create config
            config = kwargs.get("config")
            if config is None:
                config = {}
                kwargs["config"] = config
            
            # Ensure config is a dict
            if not isinstance(config, dict):
                return wrapped(*args, **kwargs)
            
            # Get or create callbacks list
            callbacks = config.get("callbacks")
            if callbacks is None:
                callbacks = []
            elif not isinstance(callbacks, list):
                callbacks = [callbacks]
            else:
                # Make a copy to avoid modifying the original
                callbacks = list(callbacks)
            
            # Avoid duplicate handlers
            if not any(isinstance(c, IAgentOpsCallbackHandler) for c in callbacks):
                callbacks.append(handler)
                config["callbacks"] = callbacks
            
            return wrapped(*args, **kwargs)

        return wrapper