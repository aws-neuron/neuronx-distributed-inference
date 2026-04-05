from unittest.mock import Mock, patch
import torch

from neuronx_distributed_inference.models.qwen2_vl.utils.input_processor import prepare_generation_inputs_hf
from neuronx_distributed_inference.models.qwen2_vl.utils.constants import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import NeuronQwen2VLForCausalLM


class TestPrepareGenerationInputsHf:

    def test_text_only_input(self):
        """Test with text prompt only, no images"""
        mock_processor = Mock()
        mock_processor.apply_chat_template.return_value = "processed_text"
        expected_tensor = torch.tensor([[1, 2, 3]])
        mock_processor.return_value = {"input_ids": expected_tensor}

        result = prepare_generation_inputs_hf(
            text_prompt="Hello world",
            image_paths=None,
            hf_qwen2_vl_processor=mock_processor
        )

        expected_messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello world"}]
        }]
        mock_processor.apply_chat_template.assert_called_once_with(
            expected_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        mock_processor.assert_called_once_with(
            text=["processed_text"],
            images=None,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        assert "input_ids" in result
        assert torch.equal(result["input_ids"], expected_tensor)

    @patch('neuronx_distributed_inference.models.qwen2_vl.utils.input_processor.Image')
    def test_text_image_input(self, mock_image_class):
        """Test with text prompt and single image"""
        mock_image = Mock()
        mock_image.resize.return_value = mock_image
        mock_image_class.open.return_value = mock_image

        mock_processor = Mock()
        mock_processor.apply_chat_template.return_value = "processed_text"
        expected_tensor = torch.tensor([[1, 2, 3]])
        mock_processor.return_value = {"input_ids": expected_tensor}

        result = prepare_generation_inputs_hf(
            text_prompt="Describe this image",
            image_paths=["image.jpg"],
            hf_qwen2_vl_processor=mock_processor
        )

        mock_image_class.open.assert_called_once_with("image.jpg")
        mock_image.resize.assert_called_once_with((DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT))

        expected_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image", "resized_height": DEFAULT_IMAGE_HEIGHT, "resized_width": DEFAULT_IMAGE_WIDTH}
            ]
        }]
        mock_processor.apply_chat_template.assert_called_once_with(
            expected_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        mock_processor.assert_called_once_with(
            text=["processed_text"],
            images=[mock_image],
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        assert "input_ids" in result
        assert torch.equal(result["input_ids"], expected_tensor)

    @patch('neuronx_distributed_inference.models.qwen2_vl.utils.input_processor.Image')
    def test_config_image_dimensions(self, mock_image_class):
        """Test with config providing custom image dimensions"""
        mock_image = Mock()
        mock_image.resize.return_value = mock_image
        mock_image_class.open.return_value = mock_image

        mock_config = Mock()
        mock_config.vision_config.neuron_config.default_image_width = 512
        mock_config.vision_config.neuron_config.default_image_height = 384

        mock_processor = Mock()
        mock_processor.apply_chat_template.return_value = "processed_text"
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        prepare_generation_inputs_hf(
            text_prompt="Test image",
            image_paths=["/path/to/image.jpg"],
            hf_qwen2_vl_processor=mock_processor,
            config=mock_config
        )

        mock_image.resize.assert_called_once_with((512, 384))
        expected_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Test image"},
                {"type": "image", "resized_height": 384, "resized_width": 512}
            ]
        }]
        mock_processor.apply_chat_template.assert_called_once_with(
            expected_messages,
            tokenize=False,
            add_generation_prompt=True
        )


class TestPrepareInputArgs:

    @patch('neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision.prepare_generation_inputs_hf')
    def test_single_prompt_no_images(self, mock_prepare_inputs):
        """Test prepare_input_args with single prompt and no images"""
        expected_input_ids = torch.tensor([[1, 2, 3]])
        expected_attention_mask = torch.tensor([[1, 1, 1]])

        mock_inputs = Mock(spec=['input_ids', 'attention_mask'])
        mock_inputs.input_ids = expected_input_ids
        mock_inputs.attention_mask = expected_attention_mask
        mock_prepare_inputs.return_value = mock_inputs

        mock_processor = Mock()

        input_ids, attention_mask, vision_inputs = NeuronQwen2VLForCausalLM.prepare_input_args(
            prompts=["Hello world"],
            images=None,
            processor=mock_processor
        )

        mock_prepare_inputs.assert_called_once_with("Hello world", None, mock_processor, "user", None)
        assert torch.equal(input_ids, expected_input_ids)
        assert torch.equal(attention_mask, expected_attention_mask)
        assert vision_inputs is None

    @patch('neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision.prepare_generation_inputs_hf')
    def test_single_prompt_with_images(self, mock_prepare_inputs):
        """Test prepare_input_args with single prompt and images"""
        expected_input_ids = torch.tensor([[1, 2, 3]])
        expected_attention_mask = torch.tensor([[1, 1, 1]])
        expected_pixel_values = torch.randn(1, 3, 224, 224)
        expected_image_grid_thw = torch.tensor([[1, 2, 3]])

        mock_inputs = Mock()
        mock_inputs.input_ids = expected_input_ids
        mock_inputs.attention_mask = expected_attention_mask
        mock_inputs.pixel_values = expected_pixel_values
        mock_inputs.image_grid_thw = expected_image_grid_thw
        mock_prepare_inputs.return_value = mock_inputs

        mock_processor = Mock()
        images = ["image.jpg"]

        input_ids, attention_mask, vision_inputs = NeuronQwen2VLForCausalLM.prepare_input_args(
            prompts=["Describe image"],
            images=[images],
            processor=mock_processor
        )

        mock_prepare_inputs.assert_called_once_with("Describe image", images, mock_processor, "user", None)
        assert torch.equal(input_ids, expected_input_ids)
        assert torch.equal(attention_mask, expected_attention_mask)
        assert vision_inputs is not None
        assert torch.equal(vision_inputs["pixel_values"], expected_pixel_values)
        assert torch.equal(vision_inputs["image_grid_thw"], expected_image_grid_thw)
