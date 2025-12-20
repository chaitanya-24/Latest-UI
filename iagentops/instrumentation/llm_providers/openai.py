import importlib.util
import logging
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers

logger = logging.getLogger(__name__)

WRAPPED_METHODS = [
    # Sync
    {"package": "openai.resources.chat.completions", "object": "Completions.create", "operation": "chat"},
    {"package": "openai.resources.embeddings", "object": "Embeddings.create", "operation": "embedding"},
    # Async
    {"package": "openai.resources.chat.completions", "object": "AsyncCompletions.create", "operation": "chat"},
    {"package": "openai.resources.embeddings", "object": "AsyncEmbeddings.create", "operation": "embedding"},
]

class OpenAIInstrumentor:
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

        if importlib.util.find_spec("openai") is None:
            return

        for m in WRAPPED_METHODS:
            try:
                if importlib.util.find_spec(m["package"].split(".")[0]) is None:
                    continue
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("operation"))
                )
                logger.info(f"Instrumented OpenAI method: {m['object']}")
            except Exception as e:
                pass

    def _wrap(self, operation):
        def wrapper(wrapped, instance, args, kwargs):
            # Detect model
            model = kwargs.get("model") or "unknown"
            span_name = f"openai.{operation} {model}"
            
            op_type = SC.GEN_AI_OPERATION_TYPE_CHAT
            if operation == "embedding":
                op_type = SC.GEN_AI_OPERATION_TYPE_EMBEDDING

            with self.tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute("service.name", self.service_name)
                span.set_attribute(SC.GEN_AI_SYSTEM, "openai")
                span.set_attribute(SC.GEN_AI_OPERATION, op_type)
                span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model)
                span.set_attribute(SC.GEN_AI_LLM_PROVIDER, "openai")
                
                if self.agent_id:
                    span.set_attribute("agent.id", str(self.agent_id))

                # Set request params
                if "temperature" in kwargs:
                    span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE, kwargs["temperature"])
                if "max_tokens" in kwargs:
                    span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, kwargs["max_tokens"])
                    
                # Capture input
                if operation == "chat":
                    messages = kwargs.get("messages", [])
                    try:
                        span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, str(messages)[:5000])
                    except: pass
                
                try:
                    result = wrapped(*args, **kwargs)
                    
                    # Log token usage from response if available
                    input_tokens = 0
                    output_tokens = 0
                    
                    if hasattr(result, "usage") and result.usage:
                        input_tokens = result.usage.prompt_tokens
                        output_tokens = result.usage.completion_tokens
                        span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                        span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
                    
                    metrics.emit_metrics(0, "openai", input_tokens, output_tokens, model) # Latency handled by histogram in metrics if needed, or we calculate here
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
