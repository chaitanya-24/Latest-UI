import importlib.util
import logging
import time
import inspect
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers

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
            
            op_type = SC.GEN_AI_OPERATION_TYPE_CHAT

            def _log_completion(span, result, duration):
                helpers.emit_agent_telemetry(
                    span=span,
                    instance=instance,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    model=model,
                    duration=duration,
                    agent_id=self.agent_id,
                    system="anthropic",
                    operation=op_type
                )
                
                # Token usage for metrics
                input_tokens = 0
                output_tokens = 0
                if hasattr(result, "usage") and result.usage:
                    input_tokens = getattr(result.usage, "input_tokens", 0)
                    output_tokens = getattr(result.usage, "output_tokens", 0)
                
                metrics.emit_metrics(duration * 1000, "anthropic", input_tokens, output_tokens, model)
                span.set_status(Status(StatusCode.OK))

            def _log_error(span, e):
                span.set_status(Status(StatusCode.ERROR, str(e)))
                try:
                    span.set_attribute(SC.ERROR_TYPE, type(e).__name__)
                except Exception:
                    pass
                span.record_exception(e)

            # Check if it's an async call
            if inspect.iscoroutinefunction(wrapped):
                async def async_wrapper():
                    start_time = time.perf_counter()
                    with self.tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                        try:
                            result = await wrapped(*args, **kwargs)
                            _log_completion(span, result, time.perf_counter() - start_time)
                            return result
                        except Exception as e:
                            _log_error(span, e)
                            raise
                return async_wrapper()
            else:
                start_time = time.perf_counter()
                with self.tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                    try:
                        result = wrapped(*args, **kwargs)
                        _log_completion(span, result, time.perf_counter() - start_time)
                        return result
                    except Exception as e:
                        _log_error(span, e)
                        raise

        return wrapper
