"""Test resale-prop-analysis."""

import resale_prop_analysis


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(resale_prop_analysis.__name__, str)
