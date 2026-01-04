"""Utility functions for extracting JSON schemas from Python function signatures and docstrings."""

import inspect
import re
import warnings
from typing import get_origin


def _map_python_type_to_json_schema(python_type):
    """Map Python type hints to JSON Schema types.

    Args:
        python_type: Python type annotation (str, int, List[str], etc.)

    Returns:
        str: JSON Schema type string ("string", "integer", "array", etc.)
    """
    # Handle missing type hints
    if python_type == inspect.Parameter.empty:
        return "string"

    # Direct type mapping
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Check direct types first
    if python_type in type_map:
        return type_map[python_type]

    # Handle typing generics (List[str], Dict[str, int], etc.)
    origin = get_origin(python_type)
    if origin in type_map:
        return type_map[origin]

    # Default to string for unknown types
    return "string"


def _parse_google_docstring(docstring):
    """Parse Google-style docstring to extract function and parameter descriptions.

    Args:
        docstring: Function's __doc__ string in Google format

    Returns:
        tuple: (description: str, param_descriptions: dict)
            - description: Function summary from docstring
            - param_descriptions: Dict mapping parameter names to descriptions
    """
    if not docstring:
        return "", {}

    # Ensure docstring is a string
    if not isinstance(docstring, str):
        warnings.warn(
            f"Docstring must be a string, got {type(docstring).__name__}. Using empty docstring."
        )
        return "", {}

    # Clean up the docstring
    docstring = inspect.cleandoc(docstring)

    # Extract function description (everything before "Args:")
    desc_match = re.search(r'^(.*?)(?=Args:|Returns:|Raises:|\Z)', docstring, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""

    # Extract Args section
    args_match = re.search(r'Args:\s*\n(.*?)(?=Returns:|Raises:|\Z)', docstring, re.DOTALL)
    if not args_match:
        return description, {}

    args_section = args_match.group(1)

    # Parse parameter descriptions
    # Pattern: param_name: description
    param_descriptions = {}
    param_pattern = r'^\s*(\w+):\s*(.+?)(?=\n\s*\w+:|\Z)'

    for match in re.finditer(param_pattern, args_section, re.MULTILINE | re.DOTALL):
        param_name = match.group(1)
        param_desc = match.group(2).strip().replace('\n', ' ')
        param_descriptions[param_name] = param_desc

    return description, param_descriptions


def extract_function_schema(func):
    """Generate complete JSON schema from a Python function.

    Extracts parameter names, types, and descriptions from function signature
    and Google-style docstring to create an OpenAI-compatible tool schema.

    Args:
        func: Python callable to extract schema from

    Returns:
        dict: OpenAI-compatible schema with 'name', 'description', and 'parameters'

    Raises:
        TypeError: If func is not callable

    Example:
        >>> def greet(name: str, greeting: str = "Hello"):
        ...     '''Greet someone.
        ...
        ...     Args:
        ...         name: Person's name
        ...         greeting: Greeting message
        ...     '''
        ...     return f"{greeting}, {name}!"
        >>> schema = extract_function_schema(greet)
        >>> schema['name']
        'greet'
        >>> schema['parameters']['required']
        ['name']
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func)}")

    # Get function signature
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        warnings.warn(f"Cannot inspect signature of function '{func.__name__}': {e}")
        return {
            "name": func.__name__,
            "description": f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }

    # Parse docstring
    docstring = func.__doc__ or ""
    description, param_descriptions = _parse_google_docstring(docstring)

    # Warn if no docstring
    if not docstring:
        warnings.warn(
            f"Function '{func.__name__}' has no docstring. "
            f"Consider adding a Google-style docstring for better schema generation."
        )

    # Build parameter schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Map type
        json_type = _map_python_type_to_json_schema(param.annotation)

        # Warn if no type hint
        if param.annotation == inspect.Parameter.empty:
            warnings.warn(
                f"Parameter '{param_name}' in function '{func.__name__}' "
                f"has no type hint. Defaulting to 'string'."
            )

        # Build property definition
        prop = {"type": json_type}

        # Add description if available
        if param_name in param_descriptions:
            prop["description"] = param_descriptions[param_name]
        else:
            # Warn about missing description
            warnings.warn(
                f"Parameter '{param_name}' in function '{func.__name__}' "
                f"lacks description in docstring."
            )

        properties[param_name] = prop

        # Determine if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Use description from docstring, or generate default
    final_description = description or f"Call {func.__name__}"

    return {
        "name": func.__name__,
        "description": final_description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    }
