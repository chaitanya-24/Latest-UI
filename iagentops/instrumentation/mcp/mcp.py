import importlib.util
import logging
import json
import traceback
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC

logger = logging.getLogger(__name__)

WRAPPED_METHODS = [
    {"package": "mcp.client.session", "object": "ClientSession.call_tool", "operation": "tool"},
    {"package": "mcp.client.session", "object": "ClientSession.read_resource", "operation": "resource"},
    {"package": "mcp.client.session", "object": "ClientSession.list_tools", "operation": "info"},
    {"package": "mcp.client.session", "object": "ClientSession.list_resources", "operation": "info"},
    {"package": "mcp.client.session", "object": "ClientSession.list_prompts", "operation": "info"},
    {"package": "mcp.client.session", "object": "ClientSession.get_prompt", "operation": "prompt"},
]

class MCPInstrumentor:
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

        if importlib.util.find_spec("mcp") is None:
            logger.debug("MCP package not found, skipping instrumentation")
            return

        for m in WRAPPED_METHODS:
            try:
                if importlib.util.find_spec(m["package"]) is None:
                    continue
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("operation"))
                )
                logger.info(f"Instrumented MCP method: {m['object']}")
            except Exception as e:
                logger.debug(f"Failed to wrap {m['object']}: {e}")
                continue

    def _wrap(self, operation):
        def wrapper(wrapped, instance, args, kwargs):
            # Extract basic info
            span_name = f"mcp.{operation}"
            
            # Refine span name based on operation
            tool_name = None
            if operation == "tool":
                # call_tool(name: str, arguments: dict | None = None)
                if args:
                    tool_name = args[0]
                elif "name" in kwargs:
                    tool_name = kwargs["name"]
                if tool_name:
                    span_name = f"mcp.tool {tool_name}"
            elif operation == "resource":
                uri = None
                if args:
                    uri = args[0]
                elif "uri" in kwargs:
                    uri = kwargs["uri"]
                if uri:
                    span_name = f"mcp.resource {uri}"

            async def async_wrapper(*a, **k):
                # Try to get context for conversation_id etc.
                ctx = helpers.get_active_context(k)
                
                import time
                start_time = time.perf_counter()
                with self.tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                    span.set_attribute("service.name", self.service_name)
                    span.set_attribute("deployment.environment", self.environment)
                    
                    # Context Propagation
                    span.set_attribute(SC.GEN_AI_CONVERSATION_ID, ctx.get("conversation_id"))
                    span.set_attribute(SC.GEN_AI_DATA_SOURCE_ID, ctx.get("data_source_id"))
                    
                    # Set MCP-specific attributes
                    span.set_attribute(SC.GEN_AI_SYSTEM, "mcp")
                    if self.agent_id:
                        span.set_attribute(SC.AGENT_ID, str(self.agent_id))

                    if operation == "tool" and tool_name:
                        span.set_attribute(SC.GEN_AI_MCP_TOOL_NAME, tool_name)
                        # arguments
                        arguments = None
                        if len(a) > 1:
                            arguments = a[1]
                        elif "arguments" in k:
                            arguments = k["arguments"]
                        if arguments:
                            span.set_attribute(SC.GEN_AI_INPUT_MESSAGES, json.dumps(arguments, default=str))

                    try:
                        # Execute async method
                        result = await wrapped(*a, **k)
                        latency = time.perf_counter() - start_time
                        
                        # Capture metrics and other common attributes
                        helpers.emit_agent_telemetry(
                            span=span,
                            instance=instance,
                            args=a,
                            kwargs=k,
                            result=result,
                            model="mcp",
                            duration=latency,
                            agent_id=self.agent_id
                        )
                        
                        # Ensure gen_ai.system is 'mcp' for the tool call
                        span.set_attribute(SC.GEN_AI_SYSTEM, "mcp")
                    
                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        try:
                            span.set_attribute(SC.ERROR_TYPE, type(e).__name__)
                        except Exception:
                            pass
                        span.record_exception(e)
                        raise

            return async_wrapper(*args, **kwargs)

        return wrapper
