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
    {"package": "langgraph.graph.state", "object": "CompiledStateGraph.ainvoke", "operation": "workflow"},
    {"package": "langgraph.graph.state", "object": "CompiledStateGraph.astream", "operation": "workflow"},
    {"package": "langgraph.pregel", "object": "Pregel.ainvoke", "operation": "workflow"},
    {"package": "langgraph.pregel", "object": "Pregel.astream", "operation": "workflow"},
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
        async def wrapper(wrapped, instance, args, kwargs):
            # Create our callback handler
            handler = IAgentOpsCallbackHandler(
                tracer=self.tracer,
                agent_id=self.agent_id,
                service_name=self.service_name,
                environment=self.environment,
                system="langgraph"
            )
            
            # Persist framework in context
            c_ctx = kwargs.copy()
            c_ctx["framework"] = "langgraph"
            helpers.get_active_context(c_ctx)
            
            try:
                sig = inspect.signature(wrapped)
                if instance is not None:
                    bound = sig.bind(instance, *args, **kwargs)
                else:
                    bound = sig.bind(*args, **kwargs)
                
                # LangGraph methods like ainvoke(input, config=None, ...)
                config = bound.arguments.get("config")
                
                # If not found at top level, check if it's in a kwargs-like parameter
                if config is None:
                    for val in bound.arguments.values():
                        if isinstance(val, dict) and "callbacks" in val:
                            config = val
                            break

                if config is None:
                    config = {}
                    if "config" in sig.parameters:
                        bound.arguments["config"] = config
                    else:
                        for param in sig.parameters.values():
                            if param.kind == param.VAR_KEYWORD:
                                if param.name not in bound.arguments:
                                    bound.arguments[param.name] = {}
                                bound.arguments[param.name]["config"] = config
                                break

                if isinstance(config, dict):
                    callbacks = config.get("callbacks", [])
                    if callbacks is None:
                        callbacks = []
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks]
                    else:
                        callbacks = list(callbacks)
                    
                    if not any(isinstance(c, IAgentOpsCallbackHandler) for c in callbacks):
                        callbacks.append(handler)
                        config["callbacks"] = callbacks
                
                # Create a top-level span for the LangGraph execution
                span_name = f"LangGraph.{wrapped.__name__}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SC.GEN_AI_SYSTEM, "langgraph")
                    span.set_attribute(SC.AGENT_FRAMEWORK, "langgraph")
                    span.set_attribute(SC.GEN_AI_OPERATION, SC.GEN_AI_OPERATION_TYPE_WORKFLOW)
                    if self.agent_id:
                        span.set_attribute(SC.AGENT_ID, str(self.agent_id))
                    
                    start_time = time.perf_counter()
                    try:
                        result = await wrapped(*bound.args, **bound.kwargs)
                        
                        # Basic telemetry for the top-level span
                        latency = time.perf_counter() - start_time
                        span.set_attribute(SC.GEN_AI_SERVER_REQUEST_DURATION, latency)
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                
            except Exception:
                # Fallback to original call if binding fails
                return wrapped(*args, **kwargs)

        return wrapper
