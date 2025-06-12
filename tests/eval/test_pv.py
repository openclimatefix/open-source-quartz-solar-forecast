import pytest
import os
from unittest.mock import patch
from tests.utils.hf_data_utils import get_pv_data

@pytest.mark.skipif(
    not os.getenv('HF_TOKEN'),
    reason="Hugging Face token not available"
)
@pytest.mark.integration
def test_get_pv_data_integration():
    """Integration test that requires HF authentication"""
    result = get_pv_data()
    assert result is not None
    # Add more specific assertions

@patch('tests.utils.hf_data_utils.download_file_from_hf')
def test_get_pv_data_unit(mock_download, mock_metadata, mock_pv_data):
    """Unit test with mocked HuggingFace data"""
    # Configure mock to return our test data
    mock_download.side_effect = [
        mock_metadata,
        mock_pv_data
    ]
    
    result = get_pv_data()
    
    # Verify the result
    assert result is not None
    assert 'power_w' in result
    assert len(result['power_w']) == 24  # 24 hours of data
    assert len(result['site_id']) == 2   # 2 sites 