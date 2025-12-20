
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add current dir to path
sys.path.append(".")

try:
    import iagentops
    from iagentops import set_context
    print("Successfully imported iagentops and set_context")
except ImportError as e:
    print(f"Failed to import iagentops: {e}")
    sys.exit(1)

class TestSDKContext(unittest.TestCase):
    def test_context_setting(self):
        # Test that set_context doesn't crash
        try:
            set_context(conversation_id="test-conv-123", data_source_id="doc-456")
            print("set_context executed successfully")
        except Exception as e:
            self.fail(f"set_context failed: {e}")

    def test_instrumentation_registry(self):
        # Verify instrumentors are registered
        iagentops.init(disabled=True)
        from iagentops._instrumentors import INSTRUMENTORS
        print(f"Registered instrumentors: {list(INSTRUMENTORS.keys())}")
        
        required = ['crewai', 'langgraph', 'adk', 'openai', 'anthropic']
        for r in required:
            if r not in INSTRUMENTORS:
                self.fail(f"Missing instrumentor: {r}")

if __name__ == "__main__":
    unittest.main()
