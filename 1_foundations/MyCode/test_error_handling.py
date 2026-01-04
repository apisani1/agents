"""Test error handling in schema generation."""

import sys
sys.path.insert(0, '/Users/antonio/AI/MyCode/agents/1_foundations/MyCode')

from chat_factory import _convert_tools_to_openai


def valid_func(x: int):
    """Valid function.

    Args:
        x: An integer
    """
    return x


def test_error_missing_function_key():
    """Test error when dict has no 'function' key."""
    print("\n=== Test 1: Missing 'function' key ===")
    tools = [{"description": "oops, no function!"}]

    try:
        _convert_tools_to_openai(tools)
        print("❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✅ Caught ValueError: {e}")
        assert "must have 'function' key" in str(e)
        return True


def test_error_function_not_callable():
    """Test error when 'function' value is not callable."""
    print("\n=== Test 2: Function value not callable ===")
    tools = [{"function": "not_a_function"}]

    try:
        _convert_tools_to_openai(tools)
        print("❌ Should have raised TypeError")
        return False
    except TypeError as e:
        print(f"✅ Caught TypeError: {e}")
        assert "must be callable" in str(e)
        return True


def test_error_tool_not_dict_or_callable():
    """Test error when tool is neither dict nor callable."""
    print("\n=== Test 3: Tool is neither dict nor callable ===")
    tools = [123]  # Integer instead of function/dict

    try:
        _convert_tools_to_openai(tools)
        print("❌ Should have raised TypeError")
        return False
    except TypeError as e:
        print(f"✅ Caught TypeError: {e}")
        assert "must be a callable or dict" in str(e)
        return True


def test_error_function_none():
    """Test error when 'function' key exists but is None."""
    print("\n=== Test 4: Function value is None ===")
    tools = [{"function": None}]

    try:
        _convert_tools_to_openai(tools)
        print("❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✅ Caught ValueError: {e}")
        assert "must have 'function' key" in str(e)
        return True


def test_valid_cases_still_work():
    """Verify valid cases still work after adding error handling."""
    print("\n=== Test 5: Valid cases still work ===")

    # Format 1: Just function
    tools1 = [valid_func]
    openai_tools1, tool_map1 = _convert_tools_to_openai(tools1)
    assert len(openai_tools1) == 1
    print("✅ Format 1 (just function) works")

    # Format 2: Dict with auto-gen
    tools2 = [{"function": valid_func}]
    openai_tools2, tool_map2 = _convert_tools_to_openai(tools2)
    assert len(openai_tools2) == 1
    print("✅ Format 2 (dict with auto-gen) works")

    # Format 3: Dict with manual schema
    tools3 = [{
        "function": valid_func,
        "description": "Manual",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }]
    openai_tools3, tool_map3 = _convert_tools_to_openai(tools3)
    assert len(openai_tools3) == 1
    print("✅ Format 3 (dict with manual schema) works")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Error Handling Test Suite")
    print("=" * 60)

    tests = [
        test_error_missing_function_key,
        test_error_function_not_callable,
        test_error_tool_not_dict_or_callable,
        test_error_function_none,
        test_valid_cases_still_work,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✅ ALL ERROR HANDLING TESTS PASSED!")
    else:
        print(f"❌ {failed} test(s) failed")
        sys.exit(1)
