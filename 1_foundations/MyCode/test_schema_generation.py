"""Test script for schema auto-generation feature."""

import sys
import warnings

# Capture warnings
warnings.simplefilter("always")

# Add current directory to path
sys.path.insert(0, '/Users/antonio/AI/MyCode/agents/1_foundations/MyCode')

from schema_utils import extract_function_schema, _map_python_type_to_json_schema, _parse_google_docstring
from chat_factory import _convert_tools_to_openai


def test_tool(email: str, name: str = "default"):
    """Test tool function.

    Args:
        email: User email address
        name: User name

    Returns:
        dict: Test result
    """
    return {"email": email, "name": name}


def test_type_mapping():
    """Test Python type to JSON Schema type mapping."""
    print("\n=== Testing Type Mapping ===")
    assert _map_python_type_to_json_schema(str) == "string"
    assert _map_python_type_to_json_schema(int) == "integer"
    assert _map_python_type_to_json_schema(float) == "number"
    assert _map_python_type_to_json_schema(bool) == "boolean"
    assert _map_python_type_to_json_schema(list) == "array"
    assert _map_python_type_to_json_schema(dict) == "object"
    print("✅ Type mapping tests passed!")


def test_docstring_parsing():
    """Test Google-style docstring parsing."""
    print("\n=== Testing Docstring Parsing ===")
    docstring = '''Test function.

    Args:
        param1: Description 1
        param2: Description 2
    '''
    desc, params = _parse_google_docstring(docstring)
    assert desc == 'Test function.'
    assert params['param1'] == 'Description 1'
    assert params['param2'] == 'Description 2'
    print("✅ Docstring parsing tests passed!")


def test_schema_extraction():
    """Test schema extraction from function."""
    print("\n=== Testing Schema Extraction ===")
    schema = extract_function_schema(test_tool)

    print(f"Schema name: {schema['name']}")
    assert schema['name'] == 'test_tool'

    print(f"Schema description: {schema['description']}")
    assert schema['description'] == 'Test tool function.'

    print(f"Required parameters: {schema['parameters']['required']}")
    assert schema['parameters']['required'] == ['email']

    print(f"Parameter types: {[(k, v['type']) for k, v in schema['parameters']['properties'].items()]}")
    assert schema['parameters']['properties']['email']['type'] == 'string'
    assert schema['parameters']['properties']['name']['type'] == 'string'

    print("✅ Schema extraction tests passed!")


def test_format_1_just_function():
    """Test Format 1: Just passing functions."""
    print("\n=== Testing Format 1: Just Functions ===")
    tools = [test_tool]
    openai_tools, tool_map = _convert_tools_to_openai(tools)

    assert len(openai_tools) == 1
    assert openai_tools[0]['type'] == 'function'
    assert openai_tools[0]['function']['name'] == 'test_tool'
    assert 'email' in openai_tools[0]['function']['parameters']['properties']
    assert 'test_tool' in tool_map
    assert callable(tool_map['test_tool'])

    print(f"Generated schema: {openai_tools[0]['function']}")
    print("✅ Format 1 test passed!")


def test_format_2_dict_with_autogen():
    """Test Format 2: Dict with auto-generation."""
    print("\n=== Testing Format 2: Dict with Auto-gen ===")
    tools = [{"function": test_tool, "description": "Custom description override"}]
    openai_tools, tool_map = _convert_tools_to_openai(tools)

    assert len(openai_tools) == 1
    assert openai_tools[0]['function']['description'] == 'Custom description override'
    assert openai_tools[0]['function']['name'] == 'test_tool'
    assert 'email' in openai_tools[0]['function']['parameters']['properties']

    print(f"Generated schema: {openai_tools[0]['function']}")
    print("✅ Format 2 test passed!")


def test_format_3_manual_schema():
    """Test Format 3: Manual schema (backward compatible)."""
    print("\n=== Testing Format 3: Manual Schema ===")
    manual_schema = {
        "function": test_tool,
        "description": "Manual description",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Manual email desc"},
            },
            "required": ["email"],
            "additionalProperties": False
        }
    }
    tools = [manual_schema]
    openai_tools, tool_map = _convert_tools_to_openai(tools)

    assert len(openai_tools) == 1
    assert openai_tools[0]['function']['description'] == 'Manual description'
    assert openai_tools[0]['function']['parameters'] == manual_schema['parameters']

    print(f"Generated schema: {openai_tools[0]['function']}")
    print("✅ Format 3 test passed!")


def test_mixed_formats():
    """Test mixing all three formats."""
    print("\n=== Testing Mixed Formats ===")

    def another_func(x: int):
        """Another test function.

        Args:
            x: An integer

        Returns:
            int: Result
        """
        return x * 2

    manual_tool = {
        "function": test_tool,
        "description": "Manual",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }

    tools = [
        another_func,  # Format 1
        {"function": test_tool},  # Format 2
        manual_tool,  # Format 3
    ]

    openai_tools, tool_map = _convert_tools_to_openai(tools)

    assert len(openai_tools) == 3
    assert openai_tools[0]['function']['name'] == 'another_func'  # Format 1
    assert openai_tools[1]['function']['name'] == 'test_tool'  # Format 2
    assert openai_tools[2]['function']['description'] == 'Manual'  # Format 3

    print("✅ Mixed formats test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Schema Auto-Generation Test Suite")
    print("=" * 60)

    try:
        test_type_mapping()
        test_docstring_parsing()
        test_schema_extraction()
        test_format_1_just_function()
        test_format_2_dict_with_autogen()
        test_format_3_manual_schema()
        test_mixed_formats()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
