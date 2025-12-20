import pytest
import torch
from unittest.mock import Mock, patch

from neuronx_distributed_inference.experimental.core.processor.bucketing_processor import (
    collect_buckets,
    get_buckets_by_model_type,
    select_smallest_bucket,
    BucketingProcessor,
)


class TestCollectBuckets:
    """Test suite for collect_buckets function"""
    
    def test_collect_buckets_basic(self):
        """Test basic bucket collection with prefill and decode models"""
        # Arrange
        reserved_example_inputs = {
            "prefill_512": (
                torch.zeros(1, 512),
                torch.zeros(1),
                torch.zeros(1, 512)  # attention_mask
            ),
            "prefill_1024": (
                torch.zeros(1, 1024),
                torch.zeros(1),
                torch.zeros(1, 1024)  # attention_mask
            ),
            "decode_128": (
                torch.zeros(1, 1),
                torch.zeros(1),
                torch.zeros(1, 128)  # attention_mask with kv_len=128
            ),
            "decode_256": (
                torch.zeros(1, 1),
                torch.zeros(1),
                torch.zeros(1, 256)  # attention_mask with kv_len=256
            )
        }
        
        # Act
        buckets, bucket_table = collect_buckets(reserved_example_inputs)
        
        # Assert
        assert torch.equal(buckets["prefill"][512][2], torch.zeros(1, 512))
        assert torch.equal(buckets["prefill"][1024][2], torch.zeros(1, 1024))
        assert torch.equal(buckets["decode"][128][2], torch.zeros(1, 128))
        assert torch.equal(buckets["decode"][256][2], torch.zeros(1, 256))
        assert bucket_table["prefill"] == [512, 1024]
        assert bucket_table["decode"] == [128, 256]


    def test_collect_buckets_sorting(self):
        """Test that bucket_table is properly sorted"""
        # Arrange
        reserved_example_inputs = {
            "prefill_2048": (None, None, torch.zeros(1, 2048)),
            "prefill_512": (None, None, torch.zeros(1, 512)),
            "prefill_1024": (None, None, torch.zeros(1, 1024)),
            "decode_256": (None, None, torch.zeros(1, 256)),
            "decode_64": (None, None, torch.zeros(1, 64)),
            "decode_128": (None, None, torch.zeros(1, 128))
        }
        
        # Act
        _, bucket_table = collect_buckets(reserved_example_inputs)
        
        # Assert
        assert bucket_table["prefill"] == [512, 1024, 2048]
        assert bucket_table["decode"] == [64, 128, 256]


class TestGetBucketsByModelType:
    """Test suite for get_buckets_by_model_type function"""

    def setup_method(self):
        """Setup common test data"""
        self.buckets = {
            "prefill": {
                512: (torch.zeros(1, 512), None, torch.zeros(1, 512)),
                1024: (torch.zeros(1, 1024), None, torch.zeros(1, 1024))
            },
            "decode": {
                128: (torch.zeros(1, 1), None, torch.zeros(1, 128)),
                256: (torch.zeros(1, 1), None, torch.zeros(1, 256))
            }
        }
        self.bucket_table = {
            "prefill": [512, 1024],
            "decode": [128, 256]
        }

    def test_get_buckets_prefill_model(self):
        """Test bucket selection for prefill model (seq_len > 1)"""
        # Arrange
        tokens = torch.zeros(2, 100)  # batch_size=2, seq_len=100
        
        # Act
        bucket_choices, table = get_buckets_by_model_type(
            tokens, self.buckets, self.bucket_table
        )
        
        # Assert
        assert bucket_choices == self.buckets["prefill"]
        assert table == [512, 1024]

    def test_get_buckets_decode_model(self):
        """Test bucket selection for decode model (seq_len = 1)"""
        # Arrange
        tokens = torch.zeros(2, 1)  # batch_size=2, seq_len=1
        
        # Act
        bucket_choices, table = get_buckets_by_model_type(
            tokens, self.buckets, self.bucket_table
        )
        
        # Assert
        assert bucket_choices == self.buckets["decode"]
        assert table == [128, 256]


class TestSelectSmallestBucket:
    """Test suite for select_smallest_bucket function"""
    
    def setup_method(self):
        """Setup common test data"""
        self.bucket_choices = {
            128: "bucket_128",
            256: "bucket_256",
            512: "bucket_512",
            1024: "bucket_1024"
        }
        self.bucket_table = [128, 256, 512, 1024]
    
    def test_select_exact_match(self):
        """Test selection when cur_len exactly matches a bucket"""
        # Act & Assert
        assert select_smallest_bucket(self.bucket_choices, self.bucket_table, 256) == "bucket_256"
        assert select_smallest_bucket(self.bucket_choices, self.bucket_table, 1024) == "bucket_1024"

        # Test selection when cur_len falls between buckets
        assert select_smallest_bucket(self.bucket_choices, self.bucket_table, 150) == "bucket_256"
        assert select_smallest_bucket(self.bucket_choices, self.bucket_table, 257) == "bucket_512"
        assert select_smallest_bucket(self.bucket_choices, self.bucket_table, 1000) == "bucket_1024"
    
        # Test selection when cur_len is smaller than smallest bucket
        assert select_smallest_bucket(self.bucket_choices, self.bucket_table, 1) == "bucket_128"
        assert select_smallest_bucket(self.bucket_choices, self.bucket_table, 127) == "bucket_128"
    
    def test_error_when_exceeds_max_bucket(self):
        """Test that ValueError is raised when cur_len exceeds all buckets"""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            select_smallest_bucket(self.bucket_choices, self.bucket_table, 1025)
        
        assert "No valid bucket found for length 1025" in str(exc_info.value)
        assert "Available buckets: [128, 256, 512, 1024]" in str(exc_info.value)


class TestBucketingProcessor:
    """Test suite for BucketingProcessor class"""
    
    def setup_method(self):
        """Setup common test data and mocks"""
        self.mock_model = Mock()
        self.mock_model.reserved_example_inputs = {
            "prefill_512": (
                torch.zeros(1, 512),
                torch.zeros(1),
                torch.zeros(1, 512)
            ),
            "decode_128": (
                torch.zeros(1, 1),
                torch.zeros(1),
                torch.zeros(1, 128)
            )
        }
        
        self.mock_pad_token_id = 0
    
    def test_initialization(self):
        """Test BucketingProcessor initialization"""
        # Act
        processor = BucketingProcessor(self.mock_model, self.mock_pad_token_id)
        
        # Assert
        assert processor.model == self.mock_model
        assert processor.pad_token_id == self.mock_pad_token_id
        assert "prefill" in processor.buckets
        assert "decode" in processor.buckets
        assert 512 in processor.buckets["prefill"]
        assert 128 in processor.buckets["decode"]
        assert processor.bucket_table["prefill"] == [512]
        assert processor.bucket_table["decode"] == [128]
    
    def test_forward(self):
        """Test forward pass through the processor"""
        # Arrange
        processor = BucketingProcessor(self.mock_model, self.mock_pad_token_id)
        self.mock_model.return_value = "model_output"
        
        tokens = torch.zeros(1, 1)
        last_pos = torch.tensor([0])
        attention_mask = torch.zeros(1, 1)
        
        with patch.object(processor, 'pre_process') as mock_preprocess:
            mock_preprocess.return_value = (tokens, last_pos, attention_mask)
            
            # Act
            output = processor.forward(tokens, last_pos, attention_mask)
            
            # Assert
            assert output == "model_output"
            mock_preprocess.assert_called_once_with(tokens, last_pos, attention_mask)
            self.mock_model.assert_called_once_with(tokens, last_pos, attention_mask)
    
    def test_pre_process_bucket_not_found(self):
        """Test pre_process raises error when no valid bucket is found"""
        # Arrange
        processor = BucketingProcessor(self.mock_model, self.mock_pad_token_id)
        tokens = torch.zeros(1, 100)
        last_pos = torch.tensor([1000])  # Exceeds available bucket
        attention_mask = torch.zeros(1, 1001)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            processor.pre_process(tokens, last_pos, attention_mask)
        
        assert "No valid bucket found" in str(exc_info.value)
