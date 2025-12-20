import importlib.util
import logging
import json
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper
from iagentops.otel import metrics, tracing
from iagentops.semconv import SemanticConvention as SC

logger = logging.getLogger(__name__)

WRAPPED_METHODS = [
    # Chroma
    {"package": "chromadb.api.models.Collection", "object": "Collection.query", "operation": "query", "system": "chroma"},
    {"package": "chromadb.api.models.Collection", "object": "Collection.add", "operation": "add", "system": "chroma"},
    {"package": "chromadb.api.models.Collection", "object": "Collection.get", "operation": "query", "system": "chroma"},
    
    # Pinecone (GRPC and HTTP clients)
    {"package": "pinecone", "object": "Index.query", "operation": "query", "system": "pinecone"},
    {"package": "pinecone.data.index", "object": "Index.query", "operation": "query", "system": "pinecone"},
    {"package": "pinecone", "object": "Index.upsert", "operation": "add", "system": "pinecone"},
    
    # Qdrant
    {"package": "qdrant_client", "object": "QdrantClient.search", "operation": "query", "system": "qdrant"},
    {"package": "qdrant_client", "object": "QdrantClient.upsert", "operation": "add", "system": "qdrant"},

    # Faiss
    {"package": "faiss", "object": "Index.add", "operation": "add", "system": "faiss"},
    {"package": "faiss", "object": "Index.search", "operation": "query", "system": "faiss"},
    {"package": "faiss", "object": "Index.add_with_ids", "operation": "add", "system": "faiss"},
]

class VectorStoreInstrumentor:
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
        
        for m in WRAPPED_METHODS:
            try:
                # Basic check if package exists
                pkg_name = m["package"].split(".")[0]
                if importlib.util.find_spec(pkg_name) is None:
                    continue
                
                wrap_function_wrapper(
                    m["package"], m["object"], self._wrap(m.get("operation"), m.get("system"))
                )
                logger.info(f"Instrumented VectorStore method: {m['object']}")
            except Exception as e:
                # logger.debug(f"Failed to wrap {m['object']}: {e}")
                pass

    def _wrap(self, operation, system):
        def wrapper(wrapped, instance, args, kwargs):
            # Extract collection name if possible
            collection_name = "unknown"
            if system == "chroma":
                collection_name = getattr(instance, "name", "unknown")
            elif system == "qdrant":
                if args: collection_name = args[0]
                elif "collection_name" in kwargs: collection_name = kwargs["collection_name"]
            elif system == "faiss":
                # Faiss indices don't have built-in names, use type or ntotal
                collection_name = f"faiss_index_{instance.__class__.__name__}"


            span_name = f"{system}.{operation} {collection_name}"
            
            with self.tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute("service.name", self.service_name)
                span.set_attribute(SC.DB_SYSTEM, system)
                span.set_attribute(SC.DB_OPERATION, operation)
                span.set_attribute(SC.DB_COLLECTION_NAME, collection_name)
                
                # Input parameters (Top K, etc.)
                if operation == "query":
                    top_k = kwargs.get("n_results") or kwargs.get("top_k") or kwargs.get("k") or kwargs.get("limit")
                    if system == "faiss" and len(args) > 1:
                        top_k = args[1] # search(x, k)
                    
                    if top_k is not None:
                        span.set_attribute(SC.DB_VECTOR_QUERY_TOP_K, top_k)
                
                try:
                    result = wrapped(*args, **kwargs)
                    
                    # Capture result count
                    count = 0
                    if system == "chroma":
                        # Chroma returns dict with 'ids', 'distances', etc.
                        if isinstance(result, dict) and "ids" in result:
                            ids = result["ids"]
                            if isinstance(ids, list):
                                count = len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
                    elif system == "pinecone" and hasattr(result, "matches"):
                        count = len(result.matches)
                    elif system == "faiss" and operation == "query":
                        # Res is (D, I)
                        if isinstance(result, tuple) and len(result) == 2:
                             # I is shape (n_queries, k)
                             I = result[1]
                             if hasattr(I, "size"):
                                 count = I.size
                    
                    span.set_attribute(SC.DB_VECTOR_COUNT, count)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
