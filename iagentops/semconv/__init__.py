"""OpenTelemetry GenAI Semantic Conventions (subset) for iAgentOps.

Centralized constants to standardize span attributes, event names, and metric names
across frameworks and providers.
"""

class SemanticConvention:
    # General
    SERVER_PORT = "server.port"
    SERVER_ADDRESS = "server.address"
    ERROR_TYPE = "error.type"
    GEN_AI_SERVER_REQUEST_DURATION = "gen_ai.server.request.duration"
    GEN_AI_SERVER_TIME_PER_OUTPUT_TOKEN = "gen_ai.server.time_per_output_token"
    GEN_AI_SERVER_TIME_TO_FIRST_TOKEN = "gen_ai.server.time_to_first_token"
    GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS = "gen_ai.client.inference.operation.details"
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

    # GenAI metric names
    GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
    GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"

    # GenAI events
    GEN_AI_USER_MESSAGE = "gen_ai.user.message"
    GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
    GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    GEN_AI_TOOL_MESSAGE = "gen_ai.tools.message"

    # GenAI request attributes
    GEN_AI_OPERATION = "gen_ai.operation.name"
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
    GEN_AI_ENDPOINT = "gen_ai.endpoint"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_LLM="gen_ai.model.name"
    GEN_AI_LLM_PROVIDER = "gen_ai.model.provider"
    GEN_AI_REQUEST_MODEL_VERSION="gen_ai.model.version"

    # GenAI response attributes
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Additional GenAI attributes
    GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
    GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
    GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    GEN_AI_AGENT_NAME = "gen_ai.agent.name"
    GEN_AI_TASK_ID = "gen_ai.task.id"
    GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_TYPE = "gen_ai.tool.type"

    # MCP
    GEN_AI_MCP_TRANSPORT = "gen_ai.mcp.transport"
    GEN_AI_MCP_MESSAGE_ID = "gen_ai.mcp.message_id"
    GEN_AI_MCP_SERVER_NAME = "gen_ai.mcp.server.name"
    GEN_AI_MCP_TOOL_NAME = "gen_ai.mcp.tool_name"
    GEN_AI_MCP_RESOURCE_URI = "gen_ai.mcp.resource.uri"
    GEN_AI_MCP_PROMPT_NAME = "gen_ai.mcp.prompt.name"

    # Vector DB
    DB_VECTOR_QUERY_TOP_K = "db.vector.query.top_k"
    DB_VECTOR_COUNT = "db.vector.count"  # Number of documents returned
    DB_SYSTEM = "db.system"          # e.g., "chroma", "pinecone"
    DB_COLLECTION_NAME = "db.collection.name"
    DB_OPERATION = "db.operation"    # "query", "add", "delete"

    # Operation types
    GEN_AI_OPERATION_TYPE_CHAT = "chat"
    GEN_AI_OPERATION_TYPE_TEXT_COMPLETION = "text_completion"
    GEN_AI_OPERATION_TYPE_EMBEDDING = "embeddings"
    GEN_AI_OPERATION_TYPE_VECTORDB = "vectordb"
    GEN_AI_OPERATION_TYPE_WORKFLOW = "workflow"

    # Systems
    GEN_AI_SYSTEM_OPENAI = "openai"
    GEN_AI_SYSTEM_ANTHROPIC = "anthropic"

    # Agent Registry attributes
    AGENT_ID = "agent.id"
    AGENT_SDK_VERSION = "agent.sdk.version"
    # Agent framework identifier (e.g., langchain, langgraph)
    AGENT_FRAMEWORK = "agent.framework"

    # Agent Registry metrics
    AGENT_REGISTRATION_SUCCESS_TOTAL = "agent.registration.success.total"
    AGENT_ACTIVE_COUNT = "agent.active_agents.count"
    AGENT_METRICS_LAST_RECEIVED_TIMESTAMP = "agent.metrics.last_received.timestamp"
    AGENT_AUDIT_EVENTS_TOTAL = "agent.audit.events.total"
    AGENT_SDK_VERSION_DISTRIBUTION = "agent.sdk.version_distribution"

    # Additional spans
    INFERENCE_SPAN = "Inference span"
    INVOKE_AGENT_SPAN = "invoke_agent ({gen_ai.agent.name})"
    CREATE_AGENT_SPAN = "create_agent ({gen_ai.agent.name})"
    EMBEDDING_SPAN = "embedding span"
    EXECUTE_TOOL_SPAN = "execute_tool ({gen_ai.tool.name})"
