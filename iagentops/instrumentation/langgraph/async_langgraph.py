import time
import importlib.util
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers
import json
import traceback
import uuid

WRAPPED_METHODS = [
    {"package": "langgraph.llms.base", "object": "BaseLLM.apredict", "provider_attr": "client_name", "operation": "inference"},
    {"package": "langgraph.llms.base", "object": "BaseLLM.agenerate", "provider_attr": "client_name", "operation": "inference"},
    {"package": "langgraph.chains.base", "object": "BaseChain.arun", "provider_attr": None, "operation": "workflow"},
    {"package": "langgraph.tools.base", "object": "BaseTool.arun", "provider_attr": None, "operation": "tool"},
    {"package": "langgraph.embeddings.base", "object": "Embeddings.aembed", "provider_attr": None, "operation": "embedding"},
]

class AsyncLangGraphInstrumentor:
    """Instrumentor for LangGraph async LLM calls."""

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
        if importlib.util.find_spec("langgraph") is None:
            return
        import logging
        logger = logging.getLogger(__name__)
        for m in WRAPPED_METHODS:
            # Only wrap if package path exists
            try:
                if importlib.util.find_spec(m["package"]) is None:
                    continue
            except Exception:
                logger.debug(f"Package not found for wrap target {m['package']}:{m['object']}")
                continue
            try:
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("provider_attr"), m.get("operation"))
                )
            except Exception:
                logger.debug(f"Skipping wrap for {m['package']}:{m['object']}", exc_info=True)

    def _wrap(self, provider_attr, operation=None):
        def wrapper(wrapped, instance, args, kwargs):
            provider = self._detect_provider(instance, provider_attr) if provider_attr else "unknown"
            model = getattr(instance, "model_name", None) or getattr(instance, "model", None)

            class_name = instance.__class__.__name__ if instance is not None else "Unknown"
            method_name = getattr(wrapped, "__name__", "call")
            
            # Default span name
            span_name = f"{class_name}.{method_name}"
            
            # Specialized span names per requirements
            if operation == "invoke_agent":
                agent_name = getattr(instance, "name", None) or "unknown"
                span_name = f"invoke_agent ({agent_name})"
            elif operation == "tool":
                tool_name = getattr(instance, "name", None) or "unknown"
                span_name = f"execute_tool ({tool_name})"
            elif operation == "create_agent":
                agent_name = getattr(instance, "name", None) or "unknown"
                span_name = f"create_agent ({agent_name})"

            op_type = SC.GEN_AI_OPERATION_TYPE_WORKFLOW if operation in (None, "workflow", "invoke_agent") else SC.GEN_AI_OPERATION_TYPE_CHAT
            if operation == "embedding":
                op_type = SC.GEN_AI_OPERATION_TYPE_EMBEDDING

            async def async_wrapper(*a, **k):
                with self.tracer.start_as_current_span(span_name) as span:
                    max_tokens = helpers._find_max_tokens(instance, kwargs)
                    temperature = helpers.temperature(instance, kwargs)
                    top_p = helpers.top_p(instance, kwargs)
                    model_version=helpers.find_model_version(instance,kwargs)

                    span.set_attribute(SC.GEN_AI_OPERATION, op_type)
                    # span.set_attribute(SC.GEN_AI_SYSTEM, provider)
                    # Mark which agent framework produced this span
                    span.set_attribute(SC.AGENT_FRAMEWORK, "langgraph")
                    _aid = getattr(self, "agent_id", None)
                    if not _aid:
                        pass # LangGraph specific extraction fallback if needed

                    if _aid is not None and not isinstance(_aid, (str, int, float, bool, bytes)):
                        _aid = str(_aid)
                    span.set_attribute("agent.id", _aid or "unknown")
                    # Service/environment and system mapping
                    span.set_attribute("service.name", getattr(self, "service_name", "iagentops"))
                    span.set_attribute("deployment.environment", getattr(self, "environment", "development"))
                    if operation in ("inference", "embedding"):
                        span.set_attribute(SC.GEN_AI_SYSTEM, provider or "unknown")
                    else:
                        span.set_attribute(SC.GEN_AI_SYSTEM, "langgraph")
                    span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model or "unknown")

                    span.set_attribute(SC.GEN_AI_LLM, model or "unknown")
                    span.set_attribute(SC.GEN_AI_LLM_PROVIDER, provider or "unknown")
                    span.set_attribute(SC.GEN_AI_PROVIDER_NAME, provider or "unknown")
                    
                    # server info
                    srv_addr = getattr(instance, "server_address", None) or getattr(instance, "base_url", None)
                    srv_port = getattr(instance, "server_port", None)
                    if srv_addr: span.set_attribute(SC.SERVER_ADDRESS, str(srv_addr))
                    if srv_port: span.set_attribute(SC.SERVER_PORT, srv_port)

                    sys_instr = helpers.extract_system_instructions(instance, k)
                    if sys_instr: span.set_attribute(SC.GEN_AI_SYSTEM_INSTRUCTIONS, sys_instr)
                    if model_version is not None:
                        span.set_attribute(SC.GEN_AI_REQUEST_MODEL_VERSION,model_version)
                    # span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model or "unknown")
                    if temperature is not None:
                        span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE,temperature)
                    if top_p is not None:
                        span.set_attribute(SC.GEN_AI_REQUEST_TOP_P,top_p)
                    if max_tokens is not None:
                        span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
                    # Additional request parameters
                    try:
                        _fp = helpers.frequency_penalty(instance, kwargs)
                        _pp = helpers.presence_penalty(instance, kwargs)
                        _st = helpers.stop_sequences(instance, kwargs)
                        _sd = helpers.seed(instance, kwargs)
                        if _fp is not None:
                            span.set_attribute(SC.GEN_AI_REQUEST_FREQUENCY_PENALTY, _fp)
                        if _pp is not None:
                            span.set_attribute(SC.GEN_AI_REQUEST_PRESENCE_PENALTY, _pp)
                        if _st is not None:
                            span.set_attribute(SC.GEN_AI_REQUEST_STOP_SEQUENCES, _st)
                        if _sd is not None:
                            span.set_attribute(SC.GEN_AI_REQUEST_SEED, _sd)
                    except Exception:
                        pass

                    # Fallback: inspect nested llm and deep llm.llm if parameters not found
                    try:
                        if any(v is None for v in [_fp, _pp, _st, _sd]):
                            llm_obj = getattr(instance, "llm", None)
                            for current in (llm_obj, getattr(llm_obj, "llm", None)):
                                if current is None:
                                    continue
                                if _fp is None:
                                    val = getattr(current, "frequency_penalty", None)
                                    if val is not None:
                                        span.set_attribute(SC.GEN_AI_REQUEST_FREQUENCY_PENALTY, val)
                                        _fp = val
                                if _pp is None:
                                    val = getattr(current, "presence_penalty", None)
                                    if val is not None:
                                        span.set_attribute(SC.GEN_AI_REQUEST_PRESENCE_PENALTY, val)
                                        _pp = val
                                if _sd is None:
                                    val = getattr(current, "seed", None)
                                    if val is not None:
                                        span.set_attribute(SC.GEN_AI_REQUEST_SEED, val)
                                        _sd = val
                                # scan dict containers
                                for container_name in ("model_kwargs", "client_kwargs", "generation_kwargs", "invocation_params", "options"):
                                    d = getattr(current, container_name, None)
                                    if isinstance(d, dict):
                                        if _fp is None and "frequency_penalty" in d:
                                            span.set_attribute(SC.GEN_AI_REQUEST_FREQUENCY_PENALTY, d["frequency_penalty"])
                                            _fp = d["frequency_penalty"]
                                        if _pp is None and "presence_penalty" in d:
                                            span.set_attribute(SC.GEN_AI_REQUEST_PRESENCE_PENALTY, d["presence_penalty"])
                                            _pp = d["presence_penalty"]
                                        if _sd is None and "seed" in d:
                                            span.set_attribute(SC.GEN_AI_REQUEST_SEED, d["seed"]) 
                                            _sd = d["seed"]
                                        if _st is None:
                                            for kname in ("stop", "stop_sequences"):
                                                if kname in d:
                                                    span.set_attribute(SC.GEN_AI_REQUEST_STOP_SEQUENCES, d[kname])
                                                    _st = d[kname]
                                                    break
                    except Exception:
                        pass

                    # If this is a tool call, add tool metadata if present on instance
                    if operation == "tool":
                        tool_name = getattr(instance, "name", None) or getattr(instance, "tool_name", None) or "unknown"
                        tool_id = getattr(instance, "id", None) or getattr(instance, "tool_id", None)
                        if tool_id is None:
                            tool_id = str(uuid.uuid4())
                        if not isinstance(tool_id, (str, int, float, bool, bytes)):
                            tool_id = str(tool_id)
                            
                        span.set_attribute(SC.GEN_AI_TOOL_CALL_ID, tool_id)
                        span.set_attribute(SC.GEN_AI_TOOL_NAME, tool_name)
                        span.set_attribute(SC.GEN_AI_TOOL_TYPE, getattr(instance, "tool_type", "function")) # default to function
                        span.set_attribute(SC.GEN_AI_TOOL_DESCRIPTION, getattr(instance, "description", ""))

                    # Conversation and data source identifiers
                    ctx = helpers.get_active_context(k)
                    span.set_attribute(SC.GEN_AI_CONVERSATION_ID, ctx.get("conversation_id", "unknown"))
                    span.set_attribute(SC.GEN_AI_DATA_SOURCE_ID, ctx.get("data_source_id", "unknown"))

                    start = time.perf_counter()
                    try:
                        result = await wrapped(*a, **k)
                        end = time.perf_counter()

                        latency_s = end - start
                        latency_ms = latency_s * 1000
                        
                        # Extract token counts using the helper
                        try:
                            input_tokens, output_tokens = helpers.extract_tokens(a, result, model)
                            
                            # Calculate time per token with safe division
                            time_per_input = (latency_s / input_tokens) if input_tokens > 0 else 0.0
                            time_per_output = (latency_s / output_tokens) if output_tokens > 0 else 0.0
                            
                            # Set span attributes
                            span.set_attribute("gen_ai.server.time_per_input_token", time_per_input)
                            span.set_attribute("gen_ai.server.time_per_output_token", time_per_output)
                            
                        except Exception as e:
                            # Set default values on error
                            span.set_attribute("gen_ai.server.time_per_input_token", 0.0)
                            span.set_attribute("gen_ai.server.time_per_output_token", 0.0)
                        
                        # Ensure these are set regardless of token extraction success
                        span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                        span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
                        
                        span.set_attribute(SC.GEN_AI_RESPONSE_MODEL, model or "unknown")
                        span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                        span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
                        span.set_attribute(SC.GEN_AI_CLIENT_OPERATION_DURATION, latency_s)
                        # Request duration metric (s)
                        span.set_attribute("gen_ai.server.request.duration", latency_s)

                        # Additional best-effort attributes
                        # Choice count
                        try:
                            choice_count = 0
                            gens = getattr(result, "generations", None)
                            if isinstance(gens, list):
                                if gens and isinstance(gens[0], list):
                                    choice_count = len(gens[0])
                                else:
                                    choice_count = len(gens)
                            elif hasattr(result, "choices") and isinstance(result.choices, list):
                                choice_count = len(result.choices)
                            else:
                                choice_count = 1 if result is not None else 0
                            span.set_attribute("gen_ai.request.choice.count", choice_count)
                        except Exception:
                            pass

                        # Encoding formats
                        try:
                            enc = k.get("response_format") or k.get("format")
                            if enc is not None:
                                span.set_attribute("gen_ai.request.encoding_formats", enc)
                        except Exception:
                            pass

                        # System instructions
                        try:
                            sys_instr = (
                                k.get("system")
                                or k.get("system_message")
                                or getattr(instance, "system_message", None)
                            )
                            if sys_instr:
                                span.set_attribute("gen_ai.system_instructions", str(sys_instr))
                        except Exception:
                            pass

                        # Emit prompt/completion events
                        try:
                            prompt = helpers.extract_input_message(a, k)
                            if prompt:
                                span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt})
                        except Exception:
                            pass
                        try:
                            completion_text = result if isinstance(result, str) else str(result)
                            if completion_text:
                                span.add_event("gen_ai.content.completion", {"gen_ai.completion": completion_text})
                        except Exception:
                            pass

                        metrics.emit_metrics(latency_ms, provider, input_tokens, output_tokens, model)
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        try:
                            span.set_attribute("errortype", type(e).__name__)
                        except Exception:
                            pass
                        span.record_exception(e)
                        raise

            return async_wrapper(*args, **kwargs)

        return wrapper

    def _detect_provider(self, instance, provider_attr):
        """Detect provider from LLM instance."""
        provider = getattr(instance, provider_attr, None)
        if provider and provider != "unknown":
            return provider.lower()
        
        class_name = instance.__class__.__name__.lower()
        if "openai" in class_name:
            return "openai"
        elif "anthropic" in class_name:
            return "anthropic"
        elif "google" in class_name:
            return "google"
        
        module = instance.__class__.__module__
        if "openai" in module:
            return "openai"
        elif "anthropic" in module:
            return "anthropic"
        elif "google" in module:
            return "google"
        
        return "unknown"
