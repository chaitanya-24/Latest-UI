import time
import logging
import json
import traceback
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from iagentops.semconv import SemanticConvention as SC
from iagentops.context import _CONTEXT_CV

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Token Counting Heuristics ---
_ENCODER_CACHE: dict = {}
_ENCODER_LOCK = threading.Lock()

def _get_encoder(model: str | None):
    """Return a cached tiktoken encoder for model or None if unavailable."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    key = (model or "cl100k_base").lower()
    with _ENCODER_LOCK:
        if key in _ENCODER_CACHE:
            return _ENCODER_CACHE[key]
        try:
            if model:
                enc = tiktoken.encoding_for_model(model)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None
        _ENCODER_CACHE[key] = enc
        return enc

def _safe_encode(text: str, model: str | None = None) -> int:
    """Return token count for text using cached tiktoken encoder when available, otherwise a heuristic."""
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback: characters / 4
    return max(1, len(text) // 4)

# --- Parameter Extraction ---

def _get_nested_attr(obj, attr):
    """Helper to get attribute from nested objects (e.g., llm.llm)"""
    if not obj: return None
    if hasattr(obj, attr): return getattr(obj, attr)
    if hasattr(obj, 'llm'): return _get_nested_attr(obj.llm, attr)
    return None

def _extract_param(instance, kwargs, possible_keys, container_keys=None):
    if container_keys is None:
        container_keys = ['generation_options', 'invocation_params', 'model_kwargs', 'options', 'generate_content_config']
    
    # 1. Direct kwargs
    for key in possible_keys:
        if key in kwargs: return kwargs[key]
        
    # 2. Nested kwargs
    for ck in container_keys:
        d = kwargs.get(ck)
        if isinstance(d, dict):
            for key in possible_keys:
                if key in d: return d[key]
                
    # 3. Instance attributes (direct or nested llm)
    for key in possible_keys:
        val = _get_nested_attr(instance, key)
        if val is not None: return val
        
    # 4. Nested instance dicts
    for ck in ['model_kwargs', 'client_kwargs', 'generation_kwargs', 'options']:
        d = getattr(instance, ck, None)
        if isinstance(d, dict):
            for key in possible_keys:
                if key in d: return d[key]
    return None

def _find_max_tokens(instance, kwargs):
    return _extract_param(instance, kwargs, ['max_tokens', 'max_output_tokens', 'max_tokens_to_sample', 'candidate_count'])

def temperature(instance, kwargs):
    return _extract_param(instance, kwargs, ['temperature'])

def top_p(instance, kwargs):
    return _extract_param(instance, kwargs, ['top_p', 'top_p_sampling'])

def frequency_penalty(instance, kwargs):
    return _extract_param(instance, kwargs, ['frequency_penalty'])

def presence_penalty(instance, kwargs):
    return _extract_param(instance, kwargs, ['presence_penalty'])

def seed(instance, kwargs):
    return _extract_param(instance, kwargs, ['seed', 'random_seed'])

def stop_sequences(instance, kwargs):
    return _extract_param(instance, kwargs, ['stop', 'stop_sequences'])

def find_model_version(instance, kwargs):
    return _extract_param(instance, kwargs, ['openai_api_version', 'api_version'])

def choice_count(instance, kwargs):
    return _extract_param(instance, kwargs, ['n', 'choice_count', 'candidate_count'])

def encoding_formats(instance, kwargs):
    return _extract_param(instance, kwargs, ['encoding_format', 'encoding_formats'])

def output_type(instance, kwargs):
    return _extract_param(instance, kwargs, ['response_format', 'output_type'])

def extract_system_instructions(instance, kwargs):
    return _extract_param(instance, kwargs, ['system', 'system_message', 'system_instructions', 'instructions'])

# --- IO Extraction ---

def extract_input_message(args, kwargs):
    """Extract user input message for telemetry from args/kwargs."""
    # 1. Check dict args for 'message', 'input', 'user_input'
    if args:
        for arg in args:
            if isinstance(arg, dict):
                for key in ['message', 'input', 'user_input']:
                    if key in arg: return arg[key]
    # 2. First string arg
    for arg in args:
        if isinstance(arg, str): return arg
    # 3. Check dict kwargs
    for v in kwargs.values():
        if isinstance(v, dict):
            for key in ['message', 'input', 'user_input']:
                if key in v: return v[key]
    # 4. Any string in kwargs
    for v in kwargs.values():
        if isinstance(v, str): return v
    return ""

def extract_tokens(args, result, model: str | None = None):
    """Compute (input_tokens, output_tokens)."""
    input_tokens = 0
    output_tokens = 0
    try:
        input_text = extract_input_message(args, {})
        input_tokens = _safe_encode(input_text, model)

        if result is not None:
            if isinstance(result, str):
                output_tokens = _safe_encode(result, model)
            elif isinstance(result, dict) and "usage" in result:
                output_tokens = int(result["usage"].get("completion_tokens", 0))
            elif hasattr(result, "content"):
                output_tokens = _safe_encode(getattr(result, "content", ""), model)
            elif hasattr(result, "raw"):
                output_tokens = _safe_encode(getattr(result, "raw", ""), model)
            elif isinstance(result, dict) and "output" in result:
                output_tokens = _safe_encode(result["output"], model)
            else:
                output_tokens = _safe_encode(str(result), model)
    except Exception:
        pass
    return int(input_tokens), int(output_tokens)

def detect_llm_provider(instance, provider_attr=None):
    """Detect provider name from instance heuristics."""
    if provider_attr:
        p = getattr(instance, provider_attr, None)
        if p and p != "unknown": return p.lower()
    
    # Check attributes
    for attr in ['provider', 'llm_provider', 'client_name']:
        val = getattr(instance, attr, None)
        if val and val != "unknown": return str(val).lower()
        
    # Heuristics based on class/module
    cls = instance.__class__.__name__.lower()
    mod = instance.__class__.__module__.lower()
    
    if "openai" in cls or "openai" in mod: return "openai"
    if "anthropic" in cls or "anthropic" in mod or "claude" in cls: return "anthropic"
    if "google" in cls or "google" in mod or "gemini" in cls or "vertex" in mod: return "google"
    if "azure" in cls or "azure" in mod: return "azure"
    
    return "unknown"

# --- Context ---

def get_active_context(kwargs):
    ctx = _CONTEXT_CV.get().copy()
    modified = False
    
    # Check kwargs first
    if "conversation_id" in kwargs: 
        ctx["conversation_id"] = kwargs["conversation_id"]
    if "data_source_id" in kwargs: 
        ctx["data_source_id"] = kwargs["data_source_id"]
    if "framework" in kwargs:
        ctx["framework"] = kwargs["framework"]
        modified = True
        
    # If still missing, generate and persist in ContextVar
    if not ctx.get("conversation_id"):
        ctx["conversation_id"] = str(uuid.uuid4())
        modified = True
    if not ctx.get("data_source_id"):
        ctx["data_source_id"] = str(uuid.uuid4())
        modified = True
        
    if modified:
        _CONTEXT_CV.set(ctx)
        
    return ctx

def _extract_agent_name(instance, kwargs):
    """Unified agent name extraction."""
    # Check instance attributes first (priority)
    for attr in ['role', 'agent_name', 'name', '_name']:
        val = getattr(instance, attr, None)
        if val: return str(val)
    
    # Check kwargs
    for key in ['agent_name', 'name', 'role']:
        if key in kwargs and kwargs[key]:
            return str(kwargs[key])
            
    return "unknown"
            
def detect_agent_framework(instance):
    """Detect agent framework from instance metadata."""
    if not instance: return "unknown"
    mod = instance.__class__.__module__.lower()
    cls = instance.__class__.__name__.lower()
    
    if "langchain" in mod: return "langchain"
    if "langgraph" in mod: return "langgraph"
    if "crewai" in mod: return "crewai"
    if "agent" in cls: return "adk"
    
    return "unknown"

# --- Main Telemetry Emitter ---

def emit_agent_telemetry(span, instance, args, kwargs, result=None, model=None, duration=None, agent_id=None, system=None):
    """Add agent attributes and prompt/completion events to the span."""
    # 1. Model
    if not model or model == "unknown":
        model = (
            getattr(instance, "model", None) or 
            getattr(instance, "model_name", None) or 
            _get_nested_attr(instance, "model") or
            _get_nested_attr(instance, "model_name") or
            "unknown"
        )
    
    # 2. Agent ID
    # Strictly use provided agent_id or instance attributes without heuristics
    if agent_id:
        _agent_id_str = str(agent_id)
    else:
        _agent_id_str = str(getattr(instance, "id", None) or getattr(instance, "agent_id", None) or "unknown")
        
    agent_name = _extract_agent_name(instance, kwargs)

    # 3. Tools
    agent_tools = getattr(instance, "tools", [])
    if isinstance(agent_tools, list):
        agent_tools = ", ".join([getattr(t, "name", str(t)) for t in agent_tools])

    # 4. Tokens & Metrics
    input_tokens, output_tokens = extract_tokens(args, result, str(model))
    token_usage = input_tokens + output_tokens
    duration = duration or 0

    # 5. Set attributes
    span.set_attribute(SC.GEN_AI_OPERATION, SC.GEN_AI_OPERATION_TYPE_WORKFLOW)
    span.set_attribute(SC.AGENT_ID, _agent_id_str)
    span.set_attribute(SC.GEN_AI_AGENT_NAME, agent_name)
    span.set_attribute("gen_ai.agent.tools", agent_tools)
    
    span.set_attribute(SC.GEN_AI_LLM, str(model))
    span.set_attribute(SC.GEN_AI_REQUEST_MODEL, str(model))
    
    provider = detect_llm_provider(instance)
    span.set_attribute(SC.GEN_AI_LLM_PROVIDER, provider)
    span.set_attribute(SC.GEN_AI_PROVIDER_NAME, provider)
    
    # Framework detection
    ctx = get_active_context(kwargs)
    framework = system or ctx.get("framework") or detect_agent_framework(instance)
    if framework != "unknown":
        span.set_attribute(SC.AGENT_FRAMEWORK, framework)
        # Only set gen_ai.system if not already set or if it's currently unknown/generic
        if framework != "adk":
             span.set_attribute(SC.GEN_AI_SYSTEM, framework)
    
    span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    span.set_attribute(SC.GEN_AI_CLIENT_TOKEN_USAGE, token_usage)
    span.set_attribute(SC.GEN_AI_CLIENT_OPERATION_DURATION, duration)
    span.set_attribute(SC.GEN_AI_SERVER_REQUEST_DURATION, duration)
    
    if input_tokens > 0:
        span.set_attribute(SC.GEN_AI_SERVER_TIME_PER_INPUT_TOKEN, duration / input_tokens)
    if output_tokens > 0:
        span.set_attribute(SC.GEN_AI_SERVER_TIME_PER_OUTPUT_TOKEN, duration / output_tokens)
    
    # Time to first token (estimate for sync calls)
    if duration > 0:
        span.set_attribute(SC.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN, duration)

    # Server Address/Port
    srv_addr = getattr(instance, "server_address", None) or getattr(instance, "base_url", None)
    srv_port = getattr(instance, "server_port", None)
    if srv_addr: span.set_attribute(SC.SERVER_ADDRESS, str(srv_addr))
    if srv_port: span.set_attribute(SC.SERVER_PORT, srv_port)

    # Params
    temp = temperature(instance, kwargs)
    if temp is not None: span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE, temp)
    tp = top_p(instance, kwargs)
    if tp is not None: span.set_attribute(SC.GEN_AI_REQUEST_TOP_P, tp)
    mt = _find_max_tokens(instance, kwargs)
    if mt is not None: span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, mt)
    
    sys_instr = extract_system_instructions(instance, kwargs)
    if sys_instr: span.set_attribute(SC.GEN_AI_SYSTEM_INSTRUCTIONS, sys_instr)

    # Missing table attributes
    cc = choice_count(instance, kwargs)
    if cc is not None: span.set_attribute(SC.GEN_AI_REQUEST_CHOICE_COUNT, cc)
    
    ef = encoding_formats(instance, kwargs)
    if ef is not None: span.set_attribute(SC.GEN_AI_REQUEST_ENCODING_FORMATS, ef)
    
    ot = output_type(instance, kwargs)
    if ot is not None: span.set_attribute(SC.GEN_AI_OUTPUT_TYPE, ot)
    
    fp = frequency_penalty(instance, kwargs)
    if fp is not None: span.set_attribute(SC.GEN_AI_REQUEST_FREQUENCY_PENALTY, fp)
    
    pp = presence_penalty(instance, kwargs)
    if pp is not None: span.set_attribute(SC.GEN_AI_REQUEST_PRESENCE_PENALTY, pp)
    
    sd = seed(instance, kwargs)
    if sd is not None: span.set_attribute(SC.GEN_AI_REQUEST_SEED, sd)
    
    st = stop_sequences(instance, kwargs)
    if st is not None: span.set_attribute(SC.GEN_AI_REQUEST_STOP_SEQUENCES, st)

    # Inference operation details (all parameters as JSON)
    try:
        details = {
            "temperature": temp,
            "top_p": tp,
            "max_tokens": mt,
            "frequency_penalty": fp,
            "presence_penalty": pp,
            "seed": sd,
            "stop": st,
            "n": cc
        }
        # Filter None
        details = {k: v for k, v in details.items() if v is not None}
        span.set_attribute(SC.GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS, json.dumps(details))
    except:
        pass

    # Context
    ctx = get_active_context(kwargs)
    span.set_attribute(SC.GEN_AI_CONVERSATION_ID, ctx.get("conversation_id"))
    span.set_attribute(SC.GEN_AI_DATA_SOURCE_ID, ctx.get("data_source_id"))
    
    # Tool ID (unique per span/operation if not provided)
    tool_id = kwargs.get("tool_id") or kwargs.get("tool_call_id") or str(uuid.uuid4())
    span.set_attribute(SC.GEN_AI_TOOL_CALL_ID, tool_id)

    # IO
    try:
        span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, json.dumps(args) if args else str(kwargs))
    except:
        span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, str(args))
        
    if result is not None:
        try:
            span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, json.dumps(result) if not isinstance(result, str) else result)
        except:
            span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, str(result))

    # Events
    now_ns = int(time.time() * 1e9)
    prompt = extract_input_message(args, kwargs)
    if prompt:
        span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": prompt}, timestamp=now_ns)
    
    completion = None
    if isinstance(result, str): completion = result
    elif isinstance(result, dict): completion = result.get("completion") or result.get("output")
    elif hasattr(result, "content"): completion = getattr(result, "content")
    
    if completion:
        span.add_event("gen_ai.content.completion", {"gen_ai.completion": str(completion)}, timestamp=now_ns)

def extract_adk_completion(result):
    if hasattr(result, "text"): return result.text
    if isinstance(result, str): return result
    if isinstance(result, list) and len(result) > 0:
        return extract_adk_completion(result[0])
    return str(result)