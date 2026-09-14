import json
import unittest
from pathlib import Path

import torch

from deeponet_darcy.model import DeepONet
from deeponet_darcy.train import make_model


class DeepONetTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = DeepONet(40, [16], [16], 8, "gelu", output_bias=False)
        result = model(torch.randn(5, 40), torch.linspace(0, 1, 17))
        self.assertEqual(tuple(result.shape), (5, 17))
        self.assertTrue(torch.isfinite(result).all())

    def test_full_config_matches_cornell_parameter_count(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "configs" / "full.json").open() as handle:
            config = json.load(handle)
        model = make_model(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 208_384)


if __name__ == "__main__":
    unittest.main()
