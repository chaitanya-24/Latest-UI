# def extract_tokens(args, result):
#     """Extract token counts from request/response"""
#     input_tokens = 0
#     output_tokens = 0
    
#     # Try to extract from result object
#     if hasattr(result, 'usage'):
#         usage = result.usage
#         input_tokens = getattr(usage, 'prompt_tokens', 0) or \
#                       getattr(usage, 'input_tokens', 0)
#         output_tokens = getattr(usage, 'completion_tokens', 0) or \
#                        getattr(usage, 'output_tokens', 0)
    
#     # Estimate from text if actual counts unavailable
#     if input_tokens == 0 and args:
#         # Rough estimation: 1 token ≈ 4 characters
#         input_text = str(args[0]) if args else ""
#         input_tokens = len(input_text) // 4
    
#     if output_tokens == 0 and hasattr(result, 'content'):
#         output_tokens = len(str(result.content)) // 4
    
#     return input_tokens, output_tokens

def extract_tokens(args, kwargs, result):
    input_tokens = 0
    output_tokens = 0

    # 1. Try to extract from result object (preferred)
    if hasattr(result, 'usage'):
        usage = result.usage
        input_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)

    # 2. Look for 'inputs' in kwargs 
    if input_tokens == 0 and 'inputs' in kwargs:
        # Often inputs is a dict with 'user_input'
        input_obj = kwargs['inputs']
        if isinstance(input_obj, dict) and 'user_input' in input_obj:
            input_text = str(input_obj['user_input'])
            input_tokens = len(input_text) // 4  

    # 3. Estimate from output if available
    if result:
            response_length = len(str(result))
            # Estimate token count (rough approximation: 4 chars per token)
            estimated_tokens = response_length // 4
            

    return input_tokens, estimated_tokens

