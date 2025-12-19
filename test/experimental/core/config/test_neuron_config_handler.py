import unittest
from omegaconf import OmegaConf
from neuronx_distributed_inference.experimental.core.config.neuron_config_handler import load_neuron_config, get_config_for_model_tag


class TestNeuronConfigHandler(unittest.TestCase):

    def setUp(self):
        self._test_cfg_path = "test/experimental/core/config/test_data/sample_neuron_config.yml"
        self._parsed_test_cfg_path = "test/experimental/core/config/test_data/parsed_sample_neuron_config.yml"
        self._expected_test_cfg = OmegaConf.load(self._parsed_test_cfg_path)

    def test_load_neuron_config(self):
        config = load_neuron_config(self._test_cfg_path)
        self.assertEqual(config, self._expected_test_cfg)

    def test_get_config_for_model_tag(self):
        model_tag_cfg = get_config_for_model_tag(self._expected_test_cfg, "prefill_1k")
        self.assertEqual(model_tag_cfg.new_module_for_prefill_1k.attribute_name, "value")
        self.assertEqual(model_tag_cfg.attention.cp_degree, 1)
    
    def test_get_config_for_model_tag_default(self):
        model_tag_cfg = get_config_for_model_tag(self._expected_test_cfg, "model_tag_without_override")
        assert "new_module_for_prefill_1k" not in model_tag_cfg
        self.assertEqual(model_tag_cfg.attention.cp_degree, 1)
        self.assertEqual(model_tag_cfg.build.batch_size, 1)

if __name__ == '__main__':
    unittest.main()
