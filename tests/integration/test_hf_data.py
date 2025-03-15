import pytest
import os
from tests.utils.hf_data_utils import get_pv_data

@pytest.mark.skipif(
    not os.getenv('HF_TOKEN'),
    reason="Hugging Face token not available"
)
@pytest.mark.integration
def test_hf_data_access():
    """Integration test for HuggingFace data access"""
    result = get_pv_data()
    assert result is not None
    # Add more specific assertions 