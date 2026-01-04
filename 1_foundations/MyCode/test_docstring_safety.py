"""Test docstring type safety."""

import sys
import warnings
sys.path.insert(0, '/Users/antonio/AI/MyCode/agents/1_foundations/MyCode')

from schema_utils import _parse_google_docstring, extract_function_schema


def test_non_string_docstring():
    """Test that non-string docstrings are handled gracefully."""
    print("\n=== Test: Non-string docstring ===")

    # Test with integer
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        desc, params = _parse_google_docstring(123)
        assert desc == ""
        assert params == {}
        assert len(w) == 1
        assert "must be a string" in str(w[0].message)
        print(f"✅ Integer docstring handled: {w[0].message}")

    # Test with list
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        desc, params = _parse_google_docstring([1, 2, 3])
        assert desc == ""
        assert params == {}
        assert len(w) == 1
        print(f"✅ List docstring handled: {w[0].message}")

    # Test with dict
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        desc, params = _parse_google_docstring({"key": "value"})
        assert desc == ""
        assert params == {}
        assert len(w) == 1
        print(f"✅ Dict docstring handled: {w[0].message}")


def test_weird_function_docstring():
    """Test function with non-string __doc__."""
    print("\n=== Test: Function with non-string __doc__ ===")

    def weird_func(x: int):
        pass

    # Manually set __doc__ to non-string (unusual but possible)
    weird_func.__doc__ = 12345

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        schema = extract_function_schema(weird_func)

        # Should still generate schema, just without description
        assert schema['name'] == 'weird_func'
        assert 'x' in schema['parameters']['properties']

        # Should have warned about non-string docstring
        warning_messages = [str(warning.message) for warning in w]
        assert any("must be a string" in msg for msg in warning_messages)
        print(f"✅ Non-string __doc__ handled gracefully")


def test_normal_docstring_still_works():
    """Test that normal docstrings still work."""
    print("\n=== Test: Normal docstrings still work ===")

    desc, params = _parse_google_docstring('''Test function.

    Args:
        x: Parameter x
        y: Parameter y
    ''')

    assert desc == 'Test function.'
    assert params['x'] == 'Parameter x'
    assert params['y'] == 'Parameter y'
    print("✅ Normal docstrings work correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Docstring Type Safety Test Suite")
    print("=" * 60)

    try:
        test_non_string_docstring()
        test_weird_function_docstring()
        test_normal_docstring_still_works()

        print("\n" + "=" * 60)
        print("✅ ALL DOCSTRING SAFETY TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
