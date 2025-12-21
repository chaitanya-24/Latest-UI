import time
import logging
import json
import traceback
from typing import Any, Dict, List, Optional
from uuid import UUID
import uuid

from opentelemetry.trace import Span, SpanKind, Status, StatusCode
import opentelemetry.trace as trace

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    class BaseCallbackHandler:
        pass
    LLMResult = Any

from iagentops.otel import tracing, metrics
from iagentops.semconv import SemanticConvention as SC
from iagentops import helpers

logger = logging.getLogger(__name__)

class IAgentOpsCallbackHandler(BaseCallbackHandler):
    """
    OpenTelemetry Callback Handler for LangChain.
    Captures traces for Chains, Tools, and LLM calls.
    """

    def __init__(self, tracer=None, agent_id=None, service_name=None, environment=None):
        self.tracer = tracer or trace.get_tracer("iagentops")
        self.agent_id = agent_id
        self.service_name = service_name
        self.environment = environment
        self.spans = {}

    def _get_span_name(self, name: str, op_type: str) -> str:
        if op_type == "tool":
            return f"execute_tool ({name})"
        if op_type == "agent" or name.lower().endswith("agent"):
            return f"invoke_agent ({name})"
        if op_type == "chat":
            return f"{name} chat"
        return f"{name} {op_type}"
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            model_name = "unknown"
            if "invocation_params" in kwargs and "model" in kwargs["invocation_params"]:
                model_name = kwargs["invocation_params"]["model"]
            elif serialized and "name" in serialized:
                model_name = serialized["name"]
            
            span_name = self._get_span_name(model_name, "chat")

            if parent_run_id and parent_run_id in self.spans:
                context = trace.set_span_in_context(self.spans[parent_run_id][0]) # Use span from tuple
                span = self.tracer.start_span(span_name, context=context, kind=SpanKind.CLIENT)
            else:
                span = self.tracer.start_span(span_name, kind=SpanKind.CLIENT)
            
            # Store span and start time
            self.spans[run_id] = (span, time.perf_counter())
            
            self._set_common_attributes(span, kwargs)
            span.set_attribute(SC.GEN_AI_OPERATION, SC.GEN_AI_OPERATION_TYPE_CHAT)
            span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(SC.GEN_AI_LLM, model_name)
            span.set_attribute(SC.GEN_AI_LLM_PROVIDER, "langchain") # Default
            span.set_attribute(SC.GEN_AI_PROVIDER_NAME, "langchain")
            span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, json.dumps(prompts) if prompts else "")
            
            # Params
            if "invocation_params" in kwargs:
                params = kwargs["invocation_params"]
                if "temperature" in params:
                    span.set_attribute(SC.GEN_AI_REQUEST_TEMPERATURE, params["temperature"])
                if "max_tokens" in params:
                    span.set_attribute(SC.GEN_AI_REQUEST_MAX_TOKENS, params["max_tokens"])

        except Exception as e:
            logger.debug(f"Error in on_llm_start: {e}")

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            if run_id not in self.spans:
                return
            span, start_time = self.spans.pop(run_id)
            
            end_time = time.perf_counter()
            latency_s = end_time - start_time
            latency_ms = latency_s * 1000
            
            span.set_attribute(SC.GEN_AI_CLIENT_OPERATION_DURATION, latency_s)
            span.set_attribute("gen_ai.server.request.duration", latency_s)

            # Usage
            input_tokens = 0
            output_tokens = 0
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            
            span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
            span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

            # Time per token
            tpi = (latency_s / input_tokens) if input_tokens > 0 else 0.0
            tpo = (latency_s / output_tokens) if output_tokens > 0 else 0.0
            span.set_attribute("gen_ai.server.time_per_input_token", tpi)
            span.set_attribute("gen_ai.server.time_per_output_token", tpo)
            
            # Completion
            if response.generations and response.generations[0]:
                completion = response.generations[0][0].text
                span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, completion[:5000])

            # Emit comprehensive telemetry
            helpers.emit_agent_telemetry(
                span=span,
                instance=None, # LangChain handler doesn't have a direct LLM instance here
                args=[],
                kwargs=kwargs,
                result=response,
                model=kwargs.get("invocation_params", {}).get("model", "unknown"),
                duration=latency_s,
                agent_id=self.agent_id
            )

            # Standard cleanup
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as e:
            logger.debug(f"Error in on_llm_end: {e}")

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            if run_id not in self.spans:
                return
            span, _ = self.spans.pop(run_id)
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
        except Exception:
            pass

    def _set_common_attributes(self, span: Span, kwargs: Dict[str, Any] = None):
        span.set_attribute("service.name", self.service_name or "iagentops")
        span.set_attribute("deployment.environment", self.environment or "development")
        if self.agent_id:
            span.set_attribute("agent.id", str(self.agent_id))
        
        # Context Propagation
        ctx = helpers.get_active_context(kwargs or {})
        span.set_attribute(SC.GEN_AI_CONVERSATION_ID, ctx.get("conversation_id", "unknown"))
        span.set_attribute(SC.GEN_AI_DATA_SOURCE_ID, ctx.get("data_source_id", "unknown"))

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            name = serialized.get("name", "Chain") if serialized else "Chain"
            span_name = self._get_span_name(name, "chain")
            
            # Start span
            if parent_run_id and parent_run_id in self.spans:
                context = trace.set_span_in_context(self.spans[parent_run_id][0])
                span = self.tracer.start_span(span_name, context=context, kind=SpanKind.INTERNAL)
            else:
                span = self.tracer.start_span(span_name, kind=SpanKind.INTERNAL)
            
            self.spans[run_id] = (span, time.perf_counter())
            
            # Attributes
            self._set_common_attributes(span, kwargs)
            span.set_attribute(SC.GEN_AI_OPERATION, SC.GEN_AI_OPERATION_TYPE_WORKFLOW)
            span.set_attribute(SC.GEN_AI_SYSTEM, "langchain")
            span.set_attribute("gen_ai.workflow.name", name)
            
            # Input
            try:
                span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, json.dumps(inputs, default=str)[:5000])
            except:
                span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, str(inputs)[:5000])

        except Exception as e:
            logger.debug(f"Error in on_chain_start: {e}")

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            if run_id not in self.spans:
                return
            span, start_time = self.spans.pop(run_id)
            
            # Output
            try:
                span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, json.dumps(outputs, default=str)[:5000])
            except:
                span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, str(outputs)[:5000])
            
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as e:
            logger.debug(f"Error in on_chain_end: {e}")

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            if run_id not in self.spans:
                return
            span, _ = self.spans.pop(run_id)
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
        except Exception as e:
            logger.debug(f"Error in on_chain_error: {e}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            name = serialized.get("name", "Tool") if serialized else "Tool"
            span_name = self._get_span_name(name, "tool")
            
            if parent_run_id and parent_run_id in self.spans:
                context = trace.set_span_in_context(self.spans[parent_run_id][0])
                span = self.tracer.start_span(span_name, context=context, kind=SpanKind.INTERNAL)
            else:
                span = self.tracer.start_span(span_name, kind=SpanKind.INTERNAL)
            
            self.spans[run_id] = (span, time.perf_counter())
            
            self._set_common_attributes(span, kwargs)
            tool_id = str(uuid.uuid4())
            span.set_attribute(SC.GEN_AI_TOOL_CALL_ID, tool_id)
            span.set_attribute(SC.GEN_AI_TOOL_NAME, name)
            span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, input_str[:5000])
            
        except Exception as e:
            logger.debug(f"Error in on_tool_start: {e}")

    def on_tool_end(
        self, output: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            if run_id not in self.spans:
                return
            span, _ = self.spans.pop(run_id)
            span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES, str(output)[:5000])
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception:
            pass

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        try:
            if run_id not in self.spans:
                return
            span, _ = self.spans.pop(run_id)
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
        except Exception:
            pass
