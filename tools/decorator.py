"""
Simplified tool decorator for LangGraph
"""
import functools
import inspect
import json
from typing import Any, Callable, Dict, Optional


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    return_direct: bool = False,
):
    """
    Decorator to convert a function into a LangGraph-compatible tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        return_direct: Whether to return the result directly to the user
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()

        # Get function signature for parameters
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"
                elif param.annotation == list:
                    param_type = "array"

            parameters[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach metadata for LangGraph
        wrapper.name = tool_name
        wrapper.description = tool_description
        wrapper.parameters = parameters
        wrapper.return_direct = return_direct
        wrapper.is_tool = True

        return wrapper

    return decorator


def convert_to_langgraph_tool(func: Callable) -> Dict[str, Any]:
    """
    Convert a tool function to LangGraph tool schema.

    Returns:
        Dict with 'name', 'description', and 'parameters' for LangGraph
    """
    if not hasattr(func, "is_tool"):
        raise ValueError(f"{func.__name__} is not a tool. Use @tool() decorator.")

    return {
        "name": func.name,
        "description": func.description,
        "parameters": func.parameters,
        "return_direct": func.return_direct,
        "function": func,
    }
