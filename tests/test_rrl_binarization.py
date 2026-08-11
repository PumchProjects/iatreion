from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch

from iatreion.configs.model_rrl import RrlConfig
from iatreion.rrl.rrl.components import BinarizeLayer
from iatreion.rrl.rrl.models import Net


class RrlCutpointConfigTest(TestCase):
    def test_defaults_to_fixed_cutpoints(self) -> None:
        config = RrlConfig(dataset=Mock(), train=Mock())

        self.assertFalse(config.trainable_cutpoints)
        self.assertEqual(config.cutpoint_tuning_eta, 0.5)

    def test_debug_name_includes_cutpoint_tuning(self) -> None:
        config = RrlConfig(
            dataset=Mock(),
            train=Mock(),
            debug=True,
            trainable_cutpoints=True,
            cutpoint_tuning_eta=0.25,
        )

        self.assertIn('_trainCutTrue_cutEta0.25', config.log_folder_name)


class BinarizeLayerCutpointTest(TestCase):
    @staticmethod
    def make_layer(
        cutpoints: torch.Tensor,
        *,
        trainable_cutpoints: bool = False,
        cutpoint_tuning_eta: float = 0.5,
    ) -> BinarizeLayer:
        with patch('torch.randn', return_value=cutpoints.clone()):
            return BinarizeLayer(
                cutpoints.shape[0],
                (0, cutpoints.shape[1]),
                trainable_cutpoints=trainable_cutpoints,
                cutpoint_tuning_eta=cutpoint_tuning_eta,
            )

    def test_fixed_mode_sorts_cutpoints_without_a_parameter(self) -> None:
        layer = self.make_layer(
            torch.tensor(
                [
                    [4.0, 10.0],
                    [0.0, 16.0],
                    [2.0, 12.0],
                ]
            )
        )

        torch.testing.assert_close(
            layer.cl,
            torch.tensor(
                [
                    [0.0, 10.0],
                    [2.0, 12.0],
                    [4.0, 16.0],
                ]
            ),
        )
        self.assertIn('base_cutpoints', dict(layer.named_buffers()))
        self.assertIn('cutpoint_radii', dict(layer.named_buffers()))
        self.assertNotIn('cutpoint_deltas', dict(layer.named_parameters()))

    def test_builds_per_column_local_movement_radii(self) -> None:
        layer = self.make_layer(
            torch.tensor(
                [
                    [0.0, 10.0],
                    [2.0, 12.0],
                    [6.0, 16.0],
                ]
            ),
            trainable_cutpoints=True,
        )

        torch.testing.assert_close(
            layer.cutpoint_radii,
            torch.tensor(
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [1.0, 1.0],
                ]
            ),
        )
        torch.testing.assert_close(layer.cl, layer.base_cutpoints)

        with torch.no_grad():
            layer.cutpoint_deltas.copy_(
                torch.tensor(
                    [
                        [100.0, -100.0],
                        [-100.0, 100.0],
                        [100.0, -100.0],
                    ]
                )
            )

        self.assertTrue(torch.all(layer.cl[1:] > layer.cl[:-1]))

    def test_optimizer_updates_cutpoints_within_their_radii(self) -> None:
        layer = self.make_layer(
            torch.tensor([[0.0], [2.0], [5.0]]),
            trainable_cutpoints=True,
        )
        optimizer = torch.optim.Adam(layer.parameters(), lr=1.0)
        values = torch.tensor([[1.0]])
        mask = torch.ones_like(values)
        initial_cutpoints = layer.cl.detach().clone()

        output, _output_mask = layer(values, mask)
        output[:, 0].sum().backward()

        self.assertIsNotNone(layer.cutpoint_deltas.grad)
        self.assertNotEqual(layer.cutpoint_deltas.grad[0, 0].item(), 0.0)
        optimizer.step()

        tuned_cutpoints = layer.cl.detach()
        self.assertFalse(torch.equal(tuned_cutpoints, initial_cutpoints))
        self.assertTrue(
            torch.all(
                torch.abs(tuned_cutpoints - layer.base_cutpoints)
                <= layer.cutpoint_radii
            )
        )
        self.assertTrue(torch.all(tuned_cutpoints[1:] > tuned_cutpoints[:-1]))

    def test_left_and_right_also_limit_movement(self) -> None:
        left = torch.tensor([0.0])
        right = torch.tensor([10.0])
        with patch(
            'torch.rand',
            return_value=torch.tensor([[0.01], [0.5], [0.99]]),
        ):
            layer = BinarizeLayer(
                3,
                (0, 1),
                left=left,
                right=right,
                trainable_cutpoints=True,
            )

        torch.testing.assert_close(
            layer.cutpoint_radii,
            torch.tensor([[0.1], [1.225], [0.1]]),
        )
        with torch.no_grad():
            layer.cutpoint_deltas.copy_(torch.tensor([[-100.0], [0.0], [100.0]]))

        self.assertTrue(torch.all(layer.cl >= left))
        self.assertTrue(torch.all(layer.cl <= right))

    def test_a_single_cutpoint_remains_fixed(self) -> None:
        layer = self.make_layer(
            torch.tensor([[1.0, 2.0]]),
            trainable_cutpoints=True,
        )
        with torch.no_grad():
            layer.cutpoint_deltas.fill_(100.0)

        torch.testing.assert_close(layer.cutpoint_radii, torch.zeros(1, 2))
        torch.testing.assert_close(layer.cl, layer.base_cutpoints)

    def test_net_state_dict_restores_tuned_cutpoints(self) -> None:
        initial = torch.tensor([[0.0], [2.0], [5.0]])
        with patch('torch.randn', return_value=initial.clone()):
            net = Net(
                [(0, 1), 3, 2],
                use_skip=False,
                trainable_cutpoints=True,
            )
        with torch.no_grad():
            net.layer_list[0].cutpoint_deltas.copy_(
                torch.tensor([[-1.0], [0.5], [1.0]])
            )
        expected = net.layer_list[0].cl.detach().clone()
        state_dict = net.state_dict()
        with patch('torch.randn', return_value=torch.zeros_like(initial)):
            restored = Net(
                [(0, 1), 3, 2],
                use_skip=False,
                trainable_cutpoints=True,
            )

        restored.load_state_dict(state_dict)

        torch.testing.assert_close(restored.layer_list[0].cl, expected)
        self.assertIn('binary1.base_cutpoints', state_dict)
        self.assertIn('binary1.cutpoint_radii', state_dict)
        self.assertIn('binary1.cutpoint_deltas', state_dict)

    def test_bound_names_use_tuned_cutpoints(self) -> None:
        layer = self.make_layer(
            torch.tensor([[0.0], [2.0]]),
            trainable_cutpoints=True,
        )
        with torch.no_grad():
            layer.cutpoint_deltas.copy_(torch.tensor([[1.0], [-1.0]]))

        layer.get_bound_name(['a'], {})

        positive_bounds = [
            float(name.rsplit(' ', 1)[1]) for name in layer.rule_name[:2]
        ]
        np.testing.assert_allclose(
            positive_bounds,
            layer.cl.detach().cpu().numpy().ravel(),
        )

    def test_rejects_an_invalid_eta(self) -> None:
        with self.assertRaisesRegex(ValueError, 'cutpoint_tuning_eta'):
            BinarizeLayer(3, (0, 1), cutpoint_tuning_eta=1.0)
