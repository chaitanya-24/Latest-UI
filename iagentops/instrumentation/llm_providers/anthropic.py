import importlib.util
import logging
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC

logger = logging.getLogger(__name__)

WRAPPED_METHODS = [
    # Sync
    {"package": "anthropic.resources.messages", "object": "Messages.create", "operation": "chat"},
    # Async
    {"package": "anthropic.resources.messages", "object": "AsyncMessages.create", "operation": "chat"},
]

class AnthropicInstrumentor:
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

        if importlib.util.find_spec("anthropic") is None:
            return

        for m in WRAPPED_METHODS:
            try:
                # Basic check
                if importlib.util.find_spec(m["package"].split(".")[0]) is None:
                    continue
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("operation"))
                )
                logger.info(f"Instrumented Anthropic method: {m['object']}")
            except Exception as e:
                pass

    def _wrap(self, operation):
        def wrapper(wrapped, instance, args, kwargs):
            model = kwargs.get("model") or "unknown"
            span_name = f"anthropic.{operation} {model}"

            with self.tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute("service.name", self.service_name)
                span.set_attribute(SC.GEN_AI_SYSTEM, "anthropic")
                span.set_attribute(SC.GEN_AI_OPERATION, SC.GEN_AI_OPERATION_TYPE_CHAT)
                span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model)
                span.set_attribute(SC.GEN_AI_LLM_PROVIDER, "anthropic")
                
                if self.agent_id:
                    span.set_attribute("agent.id", str(self.agent_id))

                if "max_tokens" in kwargs:
                    span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, kwargs["max_tokens"])
                    
                try:
                    result = wrapped(*args, **kwargs)
                    
                    input_tokens = 0
                    output_tokens = 0
                    
                    if hasattr(result, "usage"):
                        input_tokens = getattr(result.usage, "input_tokens", 0)
                        output_tokens = getattr(result.usage, "output_tokens", 0)
                        span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                        span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

                    metrics.emit_metrics(0, "anthropic", input_tokens, output_tokens, model)

                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
