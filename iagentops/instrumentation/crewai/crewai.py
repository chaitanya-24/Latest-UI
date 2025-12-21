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
    # Crew execution operations (workflow-level)
    {"package": "crewai.crew", "object": "Crew.kickoff", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.crew", "object": "Crew.kickoff_async", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.crew", "object": "Crew.kickoff_for_each", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.crew", "object": "Crew.kickoff_for_each_async", "provider_attr": None, "operation": "workflow"},

    # High-level Agent and Task operations
    {"package": "crewai.agent", "object": "Agent.execute_task", "provider_attr": None, "operation": "invoke_agent"},
    {"package": "crewai.task", "object": "Task.execute", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.task", "object": "Task.run", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.task", "object": "Task.execute_async", "provider_attr": None, "operation": "workflow"},

    # Detailed component operations
    {"package": "crewai.tools.base", "object": "BaseTool.run", "provider_attr": None, "operation": "tool"},
    {"package": "crewai.tools.base", "object": "BaseTool._run", "provider_attr": None, "operation": "tool"},
    {"package": "crewai.memory.base", "object": "BaseMemory.save", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.memory.base", "object": "BaseMemory.search", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.process", "object": "Process.kickoff", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.agent", "object": "Agent.delegate", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.agent", "object": "Agent.ask_question", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.task", "object": "Task.callback", "provider_attr": None, "operation": "workflow"},
    {"package": "crewai.task", "object": "Task._execute_core", "provider_attr": None, "operation": "workflow"},
]


def get_llm_attr(llm, key, default=None):
    # Try direct attribute
    val = getattr(llm, key, None)
    if val:
        return val
    # Try __dict__
    if hasattr(llm, "__dict__"):
        val = llm.__dict__.get(key)
        if val:
            return val
    # Try Pydantic v2 model_dump
    if hasattr(llm, "model_dump"):
        val = llm.model_dump().get(key)
        if val:
            return val
    # Try Pydantic v1 dict()
    if hasattr(llm, "dict"):
        val = llm.dict().get(key)
        if val:
            return val
    return default



def _extract_model_and_provider(instance):
    """Enhanced extraction of model and provider from instance, nested llms, agents, crew, and kwargs-like dicts if present on instance."""
    checked = set()

    def read(obj):
        if not obj or id(obj) in checked:
            return None, None
        checked.add(id(obj))

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

        # pydantic-like dumps
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
        llm = getattr(obj, "llm", None)
        if llm:
            m, p = read(llm)
            model = model or m
            provider = provider or p
            deep = getattr(llm, "llm", None)
            if deep:
                m, p = read(deep)
                model = model or m
                provider = provider or p

        # common containers
        for sub in ("agent", "parent", "crew"):
            sub_obj = getattr(obj, sub, None)
            if sub_obj:
                m, p = read(sub_obj)
                model = model or m
                provider = provider or p

        # crew agents
        agents = getattr(obj, "agents", None)
        if isinstance(agents, (list, tuple)):
            for ag in agents:
                m, p = read(ag)
                model = model or m
                provider = provider or p
                if getattr(ag, "llm", None):
                    m, p = read(ag.llm)
                    model = model or m
                    provider = provider or p

        return model, provider

    m, p = read(instance)

    # Guess provider from model
    if not p and isinstance(m, str):
        low = m.lower()
        if low.startswith("azure/"):
            p = "azure"
        elif low.startswith("openai/") or "gpt" in low:
            p = "openai"
        elif low.startswith("anthropic/") or "claude" in low:
            p = "anthropic"
        elif low.startswith("google/") or "gemini" in low:
            p = "google"

    # Fallback: helpers.detect_provider on the best llm object
    if not p:
        try:
            llm_obj = getattr(instance, "llm", None)
            if not llm_obj:
                ag = getattr(instance, "agent", None)
                llm_obj = getattr(ag, "llm", None) if ag else None
            if not llm_obj:
                agents = getattr(instance, "agents", None)
                if isinstance(agents, (list, tuple)) and agents:
                    llm_obj = getattr(agents[0], "llm", None)
            if llm_obj:
                p = helpers.detect_provider(llm_obj)
        except Exception:
            pass

    # Final fallback heuristics
    if not p:
        cls = instance.__class__.__name__.lower()
        mod = getattr(instance.__class__, "__module__", "").lower()
        if "openai" in cls or "openai" in mod or "gpt" in cls:
            p = "openai"
        elif "anthropic" in cls or "claude" in cls or "anthropic" in mod:
            p = "anthropic"
        elif "azure" in cls or "azure" in mod:
            p = "azure"
        elif "google" in cls or "gemini" in cls or "google" in mod:
            p = "google"

    return str(m) if m else "unknown", str(p) if p else "unknown"


def extract_prompt(args, kwargs, instance):
    # Try input message from args/kwargs
    prompt = helpers.extract_input_message(args, kwargs)
    # Try instance.description or .prompt if not found
    if not prompt:
        prompt = getattr(instance, "description", None) or getattr(instance, "prompt", None)
    return prompt


class CrewAIInstrumentor:
    """Instrumentor for CrewAI sync LLM calls."""

    def instrument(self, service_name="iagentops", environment="development", sdk_version="0.1.0", agent_id=None, server_address=None, server_port=None, collector_endpoint=None, use_console_exporter=True, exporter_protocol="http", **kwargs):
        import os
        os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
        
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
        # Save for span attributes
        self.service_name = service_name
        self.environment = environment
        self.agent_id = agent_id
        try:
            if importlib.util.find_spec("crewai") is None:
                return
        except Exception:
            return
        import logging
        logger = logging.getLogger(__name__)
        for m in WRAPPED_METHODS:
            try:
                if importlib.util.find_spec(m["package"]) is None:
                    continue
            except Exception:
                # Skip invalid package paths gracefully (version mismatches)
                logger.debug(f"Package not found for wrap target {m['package']}:{m['object']}")
                continue
            try:
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("provider_attr"), m.get("operation"))
                )
            except Exception:
                # Skip missing members e.g., Task.execute not present in this CrewAI version
                logger.debug(f"Skipping wrap for {m['package']}:{m['object']}", exc_info=True)


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

            # --- Robust model/provider extraction ---
            model, provider = _extract_model_and_provider(instance)

            class_name = instance.__class__.__name__ if instance is not None else "Unknown"
            method_name = getattr(wrapped, "__name__", "call")
            
            # Default span name
            span_name = f"{class_name}.{method_name}"
            
            # Specialized span names per requirements
            if operation == "invoke_agent":
                agent_name = getattr(instance, "name", None) or getattr(instance, "role", "unknown")
                span_name = f"invoke_agent ({agent_name})"
            elif operation == "tool":
                tool_name = getattr(instance, "name", None) or "unknown"
                span_name = f"execute_tool ({tool_name})"
            elif operation == "create_agent":
                agent_name = getattr(instance, "name", None) or "unknown"
                span_name = f"create_agent ({agent_name})"

            from iagentops.semconv import SemanticConvention as SC
            op_type = SC.GEN_AI_OPERATION_TYPE_WORKFLOW if operation in (None, "workflow", "invoke_agent") else SC.GEN_AI_OPERATION_TYPE_CHAT
            if operation == "embedding":
                op_type = SC.GEN_AI_OPERATION_TYPE_EMBEDDING

            with self.tracer.start_as_current_span(span_name) as span:
                # --- Explicit service/environment attributes ---
                span.set_attribute("service.name", getattr(self, "service_name", "iagentops"))
                span.set_attribute("deployment.environment", getattr(self, "environment", "development"))

                # # Agent identifier (from init, fallback to instance)
                # _agent_id_val = getattr(self, "agent_id", None) or getattr(instance, "id", None) or getattr(instance, "agent_id", None)
                # if _agent_id_val is not None and not isinstance(_agent_id_val, (str, int, float, bool, bytes)):
                #     _agent_id_val = str(_agent_id_val)
                # span.set_attribute("agent.id", _agent_id_val or "unknown")

                span.set_attribute("agent.id", _agent_id_val)

                # --- Legacy/compat attributes (optional, keep if needed) ---
                # max_tokens = helpers._find_max_tokens(instance, kwargs)
                # temperature = helpers.temperature(instance, kwargs)
                # top_p = helpers.top_p(instance, kwargs)
                model_version = helpers.find_model_version(instance, kwargs)

                span.set_attribute(SC.GEN_AI_OPERATION, op_type)
                span.set_attribute(SC.AGENT_FRAMEWORK, "crewai")
                # For CrewAI, set system to framework for workflow-level spans
                span.set_attribute(SC.GEN_AI_SYSTEM, "crewai")
                span.set_attribute("gen_ai.request.model", model)
                span.set_attribute("gen_ai.model.name", model)
                span.set_attribute("gen_ai.model.provider", provider)
                span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model)
                span.set_attribute(SC.GEN_AI_LLM, model)
                span.set_attribute(SC.GEN_AI_LLM_PROVIDER, provider)
                span.set_attribute(SC.GEN_AI_PROVIDER_NAME, provider)
                
                # server info if available on instance
                srv_addr = getattr(instance, "server_address", None) or getattr(instance, "base_url", None)
                srv_port = getattr(instance, "server_port", None)
                if srv_addr: span.set_attribute(SC.SERVER_ADDRESS, str(srv_addr))
                if srv_port: span.set_attribute(SC.SERVER_PORT, srv_port)
                
                sys_instr = helpers.extract_system_instructions(instance, kwargs)
                if sys_instr: span.set_attribute(SC.GEN_AI_SYSTEM_INSTRUCTIONS, sys_instr)

                if model_version is not None:
                    span.set_attribute(SC.GEN_AI_REQUEST_MODEL_VERSION, model_version)
                # if temperature is not None:
                #     span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE, temperature)
                # if top_p is not None:
                #     span.set_attribute(SC.GEN_AI_REQUEST_TOP_P, top_p)
                # if max_tokens is not None:
                #     span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
                # # Additional request parameters
                try:
                #     _fp = helpers.frequency_penalty(instance, kwargs)
                #     _pp = helpers.presence_penalty(instance, kwargs)
                #     _st = helpers.stop_sequences(instance, kwargs)
                #     _sd = helpers.seed(instance, kwargs)
                #     if _fp is not None:
                #         span.set_attribute(SC.GEN_AI_REQUEST_FREQUENCY_PENALTY, _fp)
                #     if _pp is not None:
                #         span.set_attribute(SC.GEN_AI_REQUEST_PRESENCE_PENALTY, _pp)
                #     if _st is not None:
                #         span.set_attribute(SC.GEN_AI_REQUEST_STOP_SEQUENCES, _st)
                #     if _sd is not None:
                #         span.set_attribute(SC.GEN_AI_REQUEST_SEED, _sd)

                     # The parameter extraction will now work with the updated helpers
                    max_tokens = helpers._find_max_tokens(instance, kwargs)
                    temperature = helpers.temperature(instance, kwargs)
                    top_p = helpers.top_p(instance, kwargs)
                    model_version = helpers.find_model_version(instance, kwargs)
                    
                    # Set parameters on span
                    if temperature is not None:
                        span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE, temperature)
                    if top_p is not None:
                        span.set_attribute(SC.GEN_AI_REQUEST_TOP_P, top_p)
                    if max_tokens is not None:
                        span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
                        
                    # Set other parameters
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

                # Best-effort fallback: inspect nested llm and deep llm.llm for missing params
                try:
                    llm_obj = getattr(instance, "llm", None)
                    for label, current in (("llm", llm_obj), ("deep", getattr(llm_obj, "llm", None))):
                        if current is None:
                            continue
                        # Direct attributes
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
                        # Dict containers to scan
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


                if operation == "tool":
                    tool_name = getattr(instance, "name", None) or getattr(instance, "tool_name", None) or "unknown"
                    tool_id = getattr(instance, "id", None) or getattr(instance, "tool_id", None)
                    if tool_id is None:
                        tool_id = str(uuid.uuid4())
                    
                    if not isinstance(tool_id, (str, int, float, bool, bytes)):
                        tool_id = str(tool_id)
                        
                    span.set_attribute(SC.GEN_AI_TOOL_CALL_ID, tool_id)
                    span.set_attribute(SC.GEN_AI_TOOL_NAME, tool_name)
                    # "gen_ai.tool.type" -> function | extension | datastore
                    span.set_attribute(SC.GEN_AI_TOOL_TYPE, getattr(instance, "tool_type", "function"))
                    span.set_attribute(SC.GEN_AI_TOOL_DESCRIPTION, getattr(instance, "description", ""))

                # Other IDs (e.g., task_id)
                if hasattr(instance, "task_id"):
                    task_id = getattr(instance, "task_id")
                    if task_id is not None and not isinstance(task_id, (str, int, float, bool, bytes)):
                        task_id = str(task_id)
                    span.set_attribute(SC.GEN_AI_TASK_ID, task_id)

                # Context propagation
                if operation != "workflow":
                     ctx = helpers.get_active_context(kwargs)
                     span.set_attribute(SC.GEN_AI_CONVERSATION_ID, ctx.get("conversation_id"))
                     span.set_attribute(SC.GEN_AI_DATA_SOURCE_ID, ctx.get("data_source_id"))

                import time
                start = time.perf_counter()
                try:
                    result = wrapped(*args, **kwargs)
                    end = time.perf_counter()
                    latency_s = end - start

                    # --- Prompt/Completion event emission ---
                    prompt = extract_prompt(args, kwargs, instance)
                    now = int(time.time() * 1e9)
                    if prompt:
                        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt}, timestamp=now)
                    if result:
                        completion = None
                        if isinstance(result, dict) and "completion" in result:
                            completion = result["completion"]
                        elif hasattr(result, "completion"):
                            completion = result.completion
                        elif isinstance(result, str):
                            completion = result
                        else:
                            completion = str(result)
                        span.add_event("gen_ai.content.completion", {"gen_ai.completion": completion}, timestamp=now)

                    # --- Agent telemetry ---
                    import iagentops.helpers as helpers_mod
                    
                    # If we have a crew ID, but no conversation_id in context, use crew ID
                    if hasattr(instance, 'crew') and instance.crew and "conversation_id" not in kwargs:
                        kwargs["conversation_id"] = str(instance.crew.id)

                    helpers_mod.emit_agent_telemetry(
                        span=span,
                        instance=instance,
                        args=args,
                        kwargs=kwargs,
                        result=result,
                        model=model,
                        duration=latency_s,
                        agent_id=getattr(self, "agent_id", None)
                    )

                    # --- 5. Metrics & Cleanup ---
                    # emit_agent_telemetry handles all 38 attributes
                    input_msg = helpers.extract_input_message(args, kwargs)
                    input_tokens = helpers._safe_encode(input_msg, model)
                    output_tokens = helpers._safe_encode(str(result), model)
                    metrics.emit_metrics(latency_s * 1000, provider, input_tokens, output_tokens, model)
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
                        "exception.stacktrace": tb,
                        "exception.escaped": "False",
                    })
                    span.record_exception(e)
                    raise

        return wrapper




    def _detect_provider(self, instance, provider_attr):
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
