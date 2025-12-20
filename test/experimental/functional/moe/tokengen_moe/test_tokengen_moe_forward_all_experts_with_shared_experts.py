import unittest
import torch
from typing import Tuple

from torch_xla.core import xla_model as xm

from neuronx_distributed_inference.experimental.functional import tokengen_moe_megakernel_forward_all_experts_with_shared_experts

torch.manual_seed(0)

# TODO: remove the @unittest.expectedFailure for tests in this file
# These tests fail due to dependency changes from the compiler.

class TestTokengenMoeMegakernelForwardWithSharedExperts(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.batch_size = 2
        self.seq_len = 64
        self.hidden_size = 256
        self.num_experts = 4
        self.intermediate_size = 512
        self.device = xm.xla_device()

    def _create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Helper method to create test inputs including shared expert weights."""
        hidden_states = torch.randn(
            self.batch_size, 
            self.seq_len, 
            self.hidden_size, 
            device=self.device
        )
        
        gamma = torch.ones(1, self.hidden_size, device=self.device)
        
        W_router = torch.randn(
            self.hidden_size, 
            self.num_experts, 
            device=self.device
        )
        
        W_expert_gate_up = torch.randn(
            self.num_experts,
            self.hidden_size,
            2,
            self.intermediate_size,
            device=self.device
        )
        
        W_expert_down = torch.randn(
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
            device=self.device
        )
        
        rank_id = torch.zeros(1, 1, device=self.device, dtype=torch.int32)
        
        # Shared expert weights
        W_shared_expert_gate = torch.randn(
            self.hidden_size,
            self.intermediate_size,
            device=self.device
        )
        
        W_shared_expert_up = torch.randn(
            self.hidden_size,
            self.intermediate_size,
            device=self.device
        )
        
        W_shared_expert_down = torch.randn(
            self.intermediate_size,
            self.hidden_size,
            device=self.device
        )
        
        inputs = [
            hidden_states, gamma, W_router, W_expert_gate_up, W_expert_down, rank_id,
            W_shared_expert_gate, W_shared_expert_up, W_shared_expert_down
        ]
        
        return tuple(inputs)

    @unittest.expectedFailure
    def test_basic_forward(self):
        """Test basic forward pass with default parameters."""
        inputs = self._create_test_inputs()
        
        output, router_logits = tokengen_moe_megakernel_forward_all_experts_with_shared_experts(*inputs)
        
        # Check output shapes
        expected_output_shape = (self.batch_size * self.seq_len, self.hidden_size)
        expected_logits_shape = (self.batch_size * self.seq_len, self.num_experts)
        
        self.assertEqual(output.shape, expected_output_shape)
        self.assertEqual(router_logits.shape, expected_logits_shape)
        
        # Check output types
        self.assertEqual(output.dtype, inputs[0].dtype)

    @unittest.expectedFailure
    def test_different_activation_functions(self):
        """Test different activation functions."""
        inputs = self._create_test_inputs()
        
        router_acts = ['sigmoid', 'softmax']
        hidden_acts = ['silu', 'gelu', 'swish']
        
        for router_act in router_acts:
            for hidden_act in hidden_acts:
                output, router_logits = tokengen_moe_megakernel_forward_all_experts_with_shared_experts(
                    *inputs,
                    router_act_fn=router_act,
                    hidden_act_fn=hidden_act
                )
                
                # Check output shapes
                expected_output_shape = (self.batch_size * self.seq_len, self.hidden_size)
                expected_logits_shape = (self.batch_size * self.seq_len, self.num_experts)
                
                self.assertEqual(output.shape, expected_output_shape)
                self.assertEqual(router_logits.shape, expected_logits_shape)
                
                # Check output types
                self.assertEqual(output.dtype, inputs[0].dtype)

    @unittest.expectedFailure
    def test_different_top_k(self):
        """Test different top_k values."""
        inputs = self._create_test_inputs()
        
        for top_k in [1, 2]:
            output, router_logits = tokengen_moe_megakernel_forward_all_experts_with_shared_experts(
                *inputs,
                top_k=top_k
            )
            
            # Check output shapes
            expected_output_shape = (self.batch_size * self.seq_len, self.hidden_size)
            expected_logits_shape = (self.batch_size * self.seq_len, self.num_experts)
            
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(router_logits.shape, expected_logits_shape)
            
            # Check output types
            self.assertEqual(output.dtype, inputs[0].dtype)

    @unittest.expectedFailure
    def test_router_pre_norm_options(self):
        """Test different router_pre_norm settings."""
        inputs = self._create_test_inputs()
        
        for router_pre_norm in [True, False]:
            output, router_logits = tokengen_moe_megakernel_forward_all_experts_with_shared_experts(
                *inputs,
                router_pre_norm=router_pre_norm
            )
            
            # Check output shapes
            expected_output_shape = (self.batch_size * self.seq_len, self.hidden_size)
            expected_logits_shape = (self.batch_size * self.seq_len, self.num_experts)
            
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(router_logits.shape, expected_logits_shape)
            
            # Check output types
            self.assertEqual(output.dtype, inputs[0].dtype)

    @unittest.expectedFailure
    def test_router_matmul_dtypes(self):
        """Test different router matrix multiplication data types."""
        inputs = self._create_test_inputs()
        
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        
        for dtype in dtypes:
            output, router_logits = tokengen_moe_megakernel_forward_all_experts_with_shared_experts(
                *inputs,
                router_matmul_dtype=dtype
            )
            
            # Check output shapes
            expected_output_shape = (self.batch_size * self.seq_len, self.hidden_size)
            expected_logits_shape = (self.batch_size * self.seq_len, self.num_experts)
            
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(router_logits.shape, expected_logits_shape)
            
            # Check output types
            self.assertEqual(output.dtype, inputs[0].dtype)

    @unittest.expectedFailure
    def test_combined_parameters(self):
        """Test with various parameter combinations."""
        inputs = self._create_test_inputs()
        
        # Test a combination of different parameters
        output, router_logits = tokengen_moe_megakernel_forward_all_experts_with_shared_experts(
            *inputs,
            eps=1e-5,
            top_k=2,
            router_act_fn='softmax',
            router_pre_norm=False,
            hidden_act_fn='gelu',
            router_matmul_dtype=torch.float32
        )
        
        # Check output shapes
        expected_output_shape = (self.batch_size * self.seq_len, self.hidden_size)
        expected_logits_shape = (self.batch_size * self.seq_len, self.num_experts)
        
        self.assertEqual(output.shape, expected_output_shape)
        self.assertEqual(router_logits.shape, expected_logits_shape)
        
        # Check output types
        self.assertEqual(output.dtype, inputs[0].dtype)


if __name__ == '__main__':
    unittest.main()
