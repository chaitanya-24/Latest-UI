import time
import importlib.util
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import tracing, metrics
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers
import json
import traceback

WRAPPED_METHODS = [
    # Crew async workflow operations
    {"package": "crewai.crew", "object": "Crew.kickoff_async", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.crew", "object": "Crew.kickoff_for_each_async", "provider_attr": None, "operation": "workflow"},

    # Task async execution
    {"package": "crewai.task", "object": "Task.execute_async", "provider_attr": None, "operation": "workflow"},
]


def _extract_model_and_provider_enhanced(instance, kwargs):
    """Best-effort extraction from instance, nested containers, and kwargs for model and provider."""
    def read_model_provider_from_obj(obj):
        model = None
        provider = None

        # direct attrs
        for mk in ("model_name", "model", "_model_name"):
            v = getattr(obj, mk, None)
            if v and not model:
                model = str(v)
        for pk in ("provider", "client_name", "api_type"):
            v = getattr(obj, pk, None)
            if v and not provider:
                provider = str(v)

        # pydantic dump/dict
        for getter in ("model_dump", "dict"):
            try:
                if hasattr(obj, getter):
                    d = getattr(obj, getter)()
                    if isinstance(d, dict):
                        if not model:
                            model = d.get("model") or d.get("model_name")
                        if not provider:
                            provider = d.get("provider") or d.get("client_name") or d.get("api_type")
            except Exception:
                pass

        # nested llm and deep llm.llm
        sub = getattr(obj, "llm", None)
        if sub:
            m2, p2 = read_model_provider_from_obj(sub)
            model = model or m2
            provider = provider or p2
            sub2 = getattr(sub, "llm", None)
            if sub2:
                m3, p3 = read_model_provider_from_obj(sub2)
                model = model or m3
                provider = provider or p3

        return model, provider

    model, provider = read_model_provider_from_obj(instance)

    # Crew-level: check agents
    try:
        agents = getattr(instance, "agents", None)
        if isinstance(agents, (list, tuple)):
            for ag in agents:
                m, p = read_model_provider_from_obj(ag)
                model = model or m
                provider = provider or p
                if getattr(ag, "llm", None):
                    m, p = read_model_provider_from_obj(ag.llm)
                    model = model or m
                    provider = provider or p
    except Exception:
        pass

    # Task-level: check instance.agent
    try:
        ag = getattr(instance, "agent", None)
        if ag:
            m, p = read_model_provider_from_obj(ag)
            model = model or m
            provider = provider or p
            if getattr(ag, "llm", None):
                m, p = read_model_provider_from_obj(ag.llm)
                model = model or m
                provider = provider or p
    except Exception:
        pass

    # kwargs hints
    for key in ("llm", "agent", "crew"):
        obj = kwargs.get(key)
        if obj:
            m, p = read_model_provider_from_obj(obj)
            model = model or m
            provider = provider or p

    # options dicts
    for k in ["generation_options", "invocation_params", "model_kwargs", "options", "config"]:
        d = kwargs.get(k)
        if isinstance(d, dict):
            if not model:
                model = str(d.get("model") or d.get("model_name") or d.get("model_id") or "") or None
            if not provider:
                provider = str(d.get("provider") or d.get("client_name") or d.get("api_type") or "") or None

    # provider guess from model string
    if not provider and isinstance(model, str):
        low = model.lower()
        if low.startswith("azure/"):
            provider = "azure"
        elif low.startswith("openai/") or "gpt" in low:
            provider = "openai"
        elif low.startswith("anthropic/") or "claude" in low:
            provider = "anthropic"
        elif low.startswith("google/") or "gemini" in low:
            provider = "google"

    # fallback to helpers.detect_provider on best candidate llm-like objects
    try:
        if not provider:
            # try instance.llm then agent.llm then first crew agent llm
            llm_obj = getattr(instance, "llm", None)
            if not llm_obj:
                ag = getattr(instance, "agent", None)
                llm_obj = getattr(ag, "llm", None) if ag else None
            if not llm_obj:
                agents = getattr(instance, "agents", None)
                if isinstance(agents, (list, tuple)) and agents:
                    llm_obj = getattr(agents[0], "llm", None)
            if llm_obj:
                provider = helpers.detect_provider(llm_obj)
    except Exception:
        pass

    # fallback to class/module heuristics
    if not provider:
        cls = instance.__class__.__name__.lower()
        mod = getattr(instance.__class__, "__module__", "").lower()
        if "openai" in cls or "openai" in mod or "gpt" in cls:
            provider = "openai"
        elif "anthropic" in cls or "claude" in cls or "anthropic" in mod:
            provider = "anthropic"
        elif "azure" in cls or "azure" in mod:
            provider = "azure"
        elif "google" in cls or "gemini" in cls or "google" in mod:
            provider = "google"

    return (model or "unknown"), (provider or "unknown")


class AsyncCrewAIInstrumentor:
    """Instrumentor for CrewAI async operations."""

    def instrument(self, service_name="iagentops", environment="development", sdk_version="0.1.0",
                   agent_id=None, server_address=None, server_port=None, collector_endpoint=None,
                   use_console_exporter=True, exporter_protocol="http", **kwargs):
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
        if importlib.util.find_spec("crewai") is None:
            return
        for m in WRAPPED_METHODS:
            if importlib.util.find_spec(m["package"]) is None:
                continue
            wrap_function_wrapper(
                m["package"], m["object"], self._wrap(m.get("provider_attr"), m.get("operation"))
            )

    def _wrap(self, provider_attr, operation=None):
        def wrapper(wrapped, instance, args, kwargs):
            # Get agent ID - Prioritize global agent ID if set
            _agent_id_val = getattr(self, "agent_id", None)
            
            if not _agent_id_val:
                _agent_id_val = (
                    getattr(instance, 'agent_id', None) or 
                    getattr(instance, 'id', None) or
                    "unknown"
                )
            
            # Ensure it's a string
            if _agent_id_val is not None and not isinstance(_agent_id_val, (str, int, float, bool, bytes)):
                _agent_id_val = str(_agent_id_val)

            model, provider = _extract_model_and_provider_enhanced(instance, kwargs)

            class_name = instance.__class__.__name__ if instance is not None else "Unknown"
            method_name = getattr(wrapped, "__name__", "call")
            span_name = f"{class_name}.{method_name}"

            async def async_wrapper(*a, **k):
                with self.tracer.start_as_current_span(span_name) as span:
                    # Operation is workflow for Crew async methods
                    span.set_attribute(SC.GEN_AI_OPERATION, SC.GEN_AI_OPERATION_TYPE_WORKFLOW)
                    span.set_attribute(SC.AGENT_FRAMEWORK, "crewai")
                    span.set_attribute("service.name", getattr(self, "service_name", "iagentops"))
                    span.set_attribute("deployment.environment", getattr(self, "environment", "development"))
                    span.set_attribute("agent.id", _agent_id_val)

                    span.set_attribute(SC.GEN_AI_SYSTEM, "crewai")
                    span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model)
                    span.set_attribute(SC.GEN_AI_LLM, model)
                    span.set_attribute(SC.GEN_AI_LLM_PROVIDER, provider)
                    
                    # Context Propagation
                    ctx = helpers.get_active_context(k)
                    span.set_attribute(SC.GEN_AI_CONVERSATION_ID, ctx.get("conversation_id", "unknown"))
                    span.set_attribute(SC.GEN_AI_DATA_SOURCE_ID, ctx.get("data_source_id", "unknown"))

                    # Extract standard params
                    try:
                        max_tokens = helpers._find_max_tokens(instance, kwargs)
                        temperature = helpers.temperature(instance, kwargs)
                        top_p = helpers.top_p(instance, kwargs)
                        model_version = helpers.find_model_version(instance, kwargs)
                        
                        if temperature is not None:
                            span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE, temperature)
                        if top_p is not None:
                            span.set_attribute(SC.GEN_AI_REQUEST_TOP_P, top_p)
                        if max_tokens is not None:
                            span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
                        if model_version is not None:
                             span.set_attribute(SC.GEN_AI_REQUEST_MODEL_VERSION, model_version)
                    except Exception:
                        pass
                    
                    start = time.perf_counter()
                    try:
                        result = await wrapped(*a, **k)
                        end = time.perf_counter()
                        latency_s = end - start
                        latency_ms = latency_s * 1000

                        # Agent telemetry
                        helpers.emit_agent_telemetry(
                            span=span,
                            instance=instance,
                            args=a,
                            kwargs=k,
                            result=result,
                            model=model,
                            duration=latency_s,
                            agent_id=getattr(self, "agent_id", None)
                        )

                        # Standard metrics
                        span.set_attribute(SC.GEN_AI_CLIENT_OPERATION_DURATION, latency_s)
                        span.set_attribute("gen_ai.server.request.duration", latency_s)

                        # Token usage
                        inp_t, out_t = helpers.extract_tokens(a, result, model)
                        span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, inp_t)
                        span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, out_t)

                        tpi = (latency_s / inp_t) if inp_t else 0.0
                        tpo = (latency_s / out_t) if out_t else 0.0
                        span.set_attribute("gen_ai.server.time_per_input_token", tpi)
                        span.set_attribute("gen_ai.server.time_per_output_token", tpo)
                        
                        # Serialize inputs/outputs for legacy support
                        try:
                            if a and len(a) >= 1:
                                span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, json.dumps(a[0]))
                            else:
                                span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, "")
                        except Exception:
                            span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, str(a if a else ""))
                        try:
                            span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, json.dumps(result))
                        except Exception:
                            span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, str(result))

                        metrics.emit_metrics(latency_ms, provider, inp_t, out_t, model)
                        return result


                        # Emit events
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

                        # Emit metrics
                        metrics.emit_metrics(latency_ms, provider, inp_t, out_t, model)
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        # Set a simple error type attribute for quick filtering
                        try:
                            span.set_attribute("errortype", type(e).__name__)
                        except Exception:
                            pass
                        tb = traceback.format_exc()
                        span.add_event("exception", {
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                            "exception.stacktrace": tb,
                            "exception.escaped": "False",
                        })
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