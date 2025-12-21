import time
import importlib.util
import json
import traceback
import uuid
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC
# from iagentops import helpers
from iagentops.instrumentation.adk import helpers

WRAPPED_METHODS = [
    {"package": "google.adk.agents", "object": "Agent.run", "provider_attr": None, "operation": "invoke_agent"},
]

class ADKInstrumentor:
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

        self._install_processor_filter()

        try:
            if importlib.util.find_spec("google.adk") is None:
                return
        except Exception:
            return
        
        for m in WRAPPED_METHODS:
            try:
                if importlib.util.find_spec(m["package"]) is None:
                    continue
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("provider_attr"), m.get("operation"))
                )
            except Exception:
                continue

    def _install_processor_filter(self):
        try:
            from opentelemetry import trace
            provider = trace.get_tracer_provider()
            processors = []
            if hasattr(provider, "_active_span_processor"):
                active = provider._active_span_processor
                if hasattr(active, "_span_processors"):
                    processors.extend(active._span_processors)
                else:
                    processors.append(active)
            for p in processors:
                original_on_end = p.on_end
                def filtering_on_end(span):
                    if span.attributes:
                        if span.attributes.get("gen_ai.system") == "gcp.vertex.agent": return 
                        for k in span.attributes.keys():
                            if k.startswith("gcp.vertex.agent"): return
                    if span.name in ["call_llm"]: return
                    if "agent_run [" in span.name: return
                    original_on_end(span)
                p.on_end = filtering_on_end
        except Exception:
            pass

    def _wrap(self, provider_attr, operation=None):
        def wrapper(wrapped, instance, args, kwargs):
            # --- 1. ID, Model & Provider ---
            _agent_id_val = getattr(self, "agent_id", None)
            if not _agent_id_val:
                _agent_id_val = (
                    getattr(instance, 'agent_id', None) or 
                    getattr(instance, 'id', None) or
                    "unknown"
                )

            if _agent_id_val is not None: _agent_id_val = str(_agent_id_val)

            model = getattr(instance, "model_name", None) or getattr(instance, "model", None)
            if hasattr(model, "model"): model = model.model
            model_str = str(model) if model else "unknown"

            provider = self._detect_provider(instance, provider_attr, model_str)
            
            class_name = instance.__class__.__name__ if instance is not None else "Unknown"
            method_name = getattr(wrapped, "__name__", "call")
            
            # Default span name
            span_name = f"{class_name}.{method_name}"
            
            # Specialized span names
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

            with self.tracer.start_as_current_span(span_name) as span:
                # --- 2. Attributes ---
                span.set_attribute("service.name", getattr(self, "service_name", "iagentops"))
                span.set_attribute("deployment.environment", getattr(self, "environment", "development"))
                span.set_attribute("agent.id", _agent_id_val)
                
                # Context Propagation
                ctx = helpers.get_active_context(kwargs)
                span.set_attribute(SC.GEN_AI_CONVERSATION_ID, ctx.get("conversation_id"))
                span.set_attribute(SC.GEN_AI_DATA_SOURCE_ID, ctx.get("data_source_id"))

                max_tokens = helpers._find_max_tokens(instance, kwargs)
                temperature = helpers.temperature(instance, kwargs)
                top_p = helpers.top_p(instance, kwargs)
                model_version = helpers.find_model_version(instance, kwargs)

                span.set_attribute(SC.GEN_AI_SYSTEM, "google-adk")
                span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model_str)
                span.set_attribute(SC.GEN_AI_LLM, model_str)
                span.set_attribute(SC.GEN_AI_LLM_PROVIDER, provider)
                span.set_attribute(SC.GEN_AI_PROVIDER_NAME, provider)
                
                # server info
                srv_addr = getattr(instance, "server_address", None) or getattr(instance, "base_url", None)
                srv_port = getattr(instance, "server_port", None)
                if srv_addr: span.set_attribute(SC.SERVER_ADDRESS, str(srv_addr))
                if srv_port: span.set_attribute(SC.SERVER_PORT, srv_port)

                sys_instr = helpers.extract_system_instructions(instance, kwargs)
                if sys_instr: span.set_attribute(SC.GEN_AI_SYSTEM_INSTRUCTIONS, sys_instr)
                span.set_attribute(SC.GEN_AI_LLM_PROVIDER, provider)
                
                if model_version: span.set_attribute(SC.GEN_AI_REQUEST_MODEL_VERSION, model_version)
                if temperature: span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE, temperature)
                if top_p: span.set_attribute(SC.GEN_AI_REQUEST_TOP_P, top_p)
                if max_tokens: span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)

                # --- 3. Execution ---
                start = time.perf_counter()
                try:
                    result = wrapped(*args, **kwargs)
                    end = time.perf_counter()
                    latency_s = end - start
                    latency_ms = latency_s * 1000

                    # --- 4. Events ---
                    prompt = helpers.extract_input_message(args, kwargs)
                    completion = helpers.extract_adk_completion(result)
                    now = int(time.time() * 1e9)

                    if prompt:
                        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt}, timestamp=now)
                    if completion:
                        span.add_event("gen_ai.content.completion", {"gen_ai.completion": completion}, timestamp=now)

                    # --- 5. Usage & IO ---
                    # --- 5. Unified Telemetry ---
                    helpers.emit_agent_telemetry(
                        span=span,
                        instance=instance,
                        args=args,
                        kwargs=kwargs,
                        result=result,
                        model=model_str,
                        duration=latency_s,
                        agent_id=self.agent_id
                    )

                    # Emit metrics manually if needed, or emit_agent_telemetry can do it
                    metrics.emit_metrics(latency_ms, provider, input_tokens, output_tokens, model_str)
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    try:
                        span.set_attribute(SC.ERROR_TYPE, type(e).__name__)
                    except Exception:
                        pass
                    tb = traceback.format_exc()
                    span.add_event("exception", {
                        "exception.type": type(e).__name__,
                        "exception.message": str(e),
                        "exception.stacktrace": tb
                    })
                    span.record_exception(e)
                    raise

        return wrapper

    def _detect_provider(self, instance, provider_attr, model_name):
        if model_name and isinstance(model_name, str):
            mn = model_name.lower()
            if "azure" in mn and "gpt" in mn: return "azure openai"
            if "azure" in mn: return "azure"
            if "gpt" in mn: return "openai"
            if "claude" in mn: return "anthropic"
            if "gemini" in mn: return "google"
        
        if provider_attr:
            p = getattr(instance, provider_attr, None)
            if p and p != "unknown": return p.lower()
        
        c = (instance.__class__.__name__).lower()
        m = (instance.__class__.__module__).lower()
        if "openai" in c or "openai" in m: return "openai"
        if "anthropic" in c or "anthropic" in m: return "anthropic"
        if "google" in c or "vertex" in m or "adk" in m: return "google"
        if "azure" in c: return "azure"
        return "unknown"