import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch
from optuna.trial import FixedTrial

from iatreion.configs.model_rrl import RrlConfig
from iatreion.rrl.binarization import (
    MAX_SHAP_SAMPLES,
    shap_jump_cutpoints,
    tabpfn_shap_cutpoints,
)
from iatreion.rrl.experiment import _get_cutpoints
from iatreion.rrl.rrl.components import (
    BinarizeLayer,
    ConjunctionLayer,
    DisjunctionLayer,
    extract_rules,
)
from iatreion.rrl.rrl.models import Net
from iatreion.runners.optuna import flatten_search_space


class RrlCutpointConfigTest(TestCase):
    def test_random_binarization_is_the_default(self) -> None:
        config = RrlConfig(dataset=Mock(), train=Mock())

        self.assertEqual(config.binarization, 'random')
        self.assertIsNone(config.tabpfn_model_path)
        self.assertEqual(config.cutpoint_tuning_eta, 0.5)
        self.assertFalse(hasattr(config, 'trainable_cutpoints'))

    def test_tabpfn_shap_requires_a_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, 'tabpfn_model_path'):
            RrlConfig(
                dataset=Mock(),
                train=Mock(),
                binarization='tabpfn-shap',
            )

    def test_debug_name_includes_binarization_and_cutpoint_tuning(self) -> None:
        config = RrlConfig(
            dataset=Mock(),
            train=Mock(),
            debug=True,
            binarization='tabpfn-shap',
            tabpfn_model_path=Path('/models/tabpfn-v3.ckpt'),
            cutpoint_tuning_eta=0.25,
        )

        self.assertIn('_bintabpfn-shap_cutEta0.25', config.log_folder_name)


class RrlCutpointOptunaTest(TestCase):
    def test_search_space_includes_the_fixed_cutpoint_baseline(self) -> None:
        path = Path(__file__).parents[1] / 'configs' / 'optuna_rrl.toml'
        with path.open('rb') as file:
            data = tomllib.load(file)
        name = 'cutpoint_tuning_eta'
        space = flatten_search_space(data['search'])[name]

        self.assertEqual(space.low, 0.0)
        self.assertEqual(space.high, 0.9)
        self.assertEqual(space.step, 0.1)
        self.assertEqual(space.sample(FixedTrial({name: 0.0}), name), 0.0)


class ShapJumpCutpointTest(TestCase):
    def test_selects_the_largest_multiclass_jump_after_averaging_duplicates(
        self,
    ) -> None:
        X = np.array(
            [
                [0.0, 5.0, 2.0],
                [0.0, 5.0, 2.0],
                [1.0, 5.0, 2.0],
                [2.0, 5.0, 2.0],
                [3.0, 5.0, np.nan],
            ]
        )
        values = np.zeros((5, 3, 2))
        values[:, 0] = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],
                [2.0, 10.0],
                [3.0, 10.0],
            ]
        )

        cutpoints = shap_jump_cutpoints(X, values, n_thresholds=1)

        np.testing.assert_allclose(cutpoints[0], [1.5])
        self.assertEqual(len(cutpoints[1]), 0)
        self.assertEqual(len(cutpoints[2]), 0)

    def test_k_is_an_upper_bound_without_quantile_filling(self) -> None:
        X = np.arange(4, dtype=float).reshape(-1, 1)
        values = np.array([0.0, 1.0, 1.0, 3.0]).reshape(-1, 1)

        cutpoints = shap_jump_cutpoints(X, values, n_thresholds=10)

        np.testing.assert_allclose(cutpoints[0], [0.5, 2.5])

    def test_constant_and_flat_features_have_no_cutpoints(self) -> None:
        X = np.column_stack([np.arange(4), np.ones(4)])
        values = np.zeros((4, 2, 2))

        cutpoints = shap_jump_cutpoints(X, values, n_thresholds=3)

        self.assertEqual([len(feature) for feature in cutpoints], [0, 0])


class TabpfnShapTest(TestCase):
    @patch('iatreion.rrl.binarization.shap.Explainer')
    @patch('iatreion.rrl.binarization._make_classifier')
    def test_fits_the_full_fold_and_explains_a_stratified_sample(
        self,
        make_classifier: Mock,
        explainer_factory: Mock,
    ) -> None:
        X = np.column_stack(
            [
                np.arange(300),
                np.arange(300) % 5,
                np.linspace(-1.0, 1.0, 300),
            ]
        )
        y = np.repeat([0, 1], 150)
        classifier = make_classifier.return_value
        explanation = SimpleNamespace(
            values=np.zeros((MAX_SHAP_SAMPLES, X.shape[1], 2))
        )
        explainer = explainer_factory.return_value
        explainer.return_value = explanation
        path = Path('/models/tabpfn-v3.ckpt')

        cutpoints = tabpfn_shap_cutpoints(
            X,
            y,
            continuous_start=1,
            n_thresholds=4,
            model_path=path,
            random_state=7,
        )

        make_classifier.assert_called_once_with(path, 7)
        np.testing.assert_array_equal(classifier.fit.call_args.args[0], X)
        np.testing.assert_array_equal(classifier.fit.call_args.args[1], y)
        X_sample = explainer_factory.call_args.args[1]
        self.assertEqual(X_sample.shape, (MAX_SHAP_SAMPLES, X.shape[1]))
        sampled_y = y[X_sample[:, 0].astype(int)]
        np.testing.assert_array_equal(np.bincount(sampled_y), [128, 128])
        self.assertEqual(explainer_factory.call_args.kwargs['algorithm'], 'permutation')
        self.assertEqual(explainer_factory.call_args.kwargs['seed'], 7)
        np.testing.assert_array_equal(explainer.call_args.args[0], X_sample)
        self.assertEqual(explainer.call_args.kwargs['max_evals'], 7)
        self.assertTrue(explainer.call_args.kwargs['silent'])
        self.assertEqual([len(feature) for feature in cutpoints], [0, 0])


class ExperimentCutpointTest(TestCase):
    @patch('iatreion.rrl.experiment.tabpfn_shap_cutpoints')
    def test_passes_only_training_data_to_tabpfn_shap(self, generate: Mock) -> None:
        X = np.arange(30, dtype=float).reshape(10, 3)
        y = np.arange(10) % 2
        generated = [np.array([1.0]), np.array([2.0, 3.0])]
        generate.return_value = generated
        path = Path('/models/tabpfn-v3.ckpt')
        args = SimpleNamespace(
            binarization='tabpfn-shap',
            tabpfn_model_path=path,
            train=SimpleNamespace(seed=11),
            use_not=True,
        )
        ctx = SimpleNamespace(
            train_data=(X, y),
            val_data=(np.full((2, 3), -1.0), np.zeros(2)),
            test_data=(np.full((2, 3), -2.0), np.zeros(2)),
            db_enc=SimpleNamespace(
                binary_flen=1,
                categorical_flen=0,
                numeric_flen=2,
                X_fname=['binary', 'a', 'b'],
            ),
        )

        result = _get_cutpoints(args, ctx, 5)

        self.assertIs(result, generated)
        np.testing.assert_array_equal(generate.call_args.args[0], X)
        np.testing.assert_array_equal(generate.call_args.args[1], y)
        self.assertEqual(
            generate.call_args.kwargs,
            {
                'continuous_start': 1,
                'n_thresholds': 5,
                'model_path': path,
                'random_state': 11,
            },
        )

    @patch('iatreion.rrl.experiment.tabpfn_shap_cutpoints')
    def test_random_mode_does_not_build_a_teacher(self, generate: Mock) -> None:
        args = SimpleNamespace(binarization='random')
        ctx = SimpleNamespace(
            db_enc=SimpleNamespace(categorical_flen=0, numeric_flen=2)
        )

        self.assertIsNone(_get_cutpoints(args, ctx, 5))
        generate.assert_not_called()


class BinarizeLayerCutpointTest(TestCase):
    def test_supports_zero_one_and_k_cutpoints_per_feature(self) -> None:
        layer = BinarizeLayer(
            2,
            (1, 3),
            cutpoints=[np.array([]), np.array([10.0]), np.array([0.0, 2.0])],
            cutpoint_tuning_eta=0.0,
        )
        values = torch.tensor([[1.0, 100.0, 15.0, 1.0]])
        mask = torch.tensor([[1.0, 0.0, 0.0, 1.0]])

        output, output_mask = layer(values, mask)

        torch.testing.assert_close(
            output,
            torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(
            output_mask,
            torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(layer.cutpoint_features, torch.tensor([1, 2, 2]))
        self.assertEqual(layer.output_dim, 7)

    def test_uses_provided_cutpoints_as_fixed_buffers_when_eta_is_zero(self) -> None:
        layer = BinarizeLayer(
            2,
            (0, 2),
            cutpoints=[np.array([1.0, 0.0]), np.array([10.0])],
            cutpoint_tuning_eta=0.0,
        )

        torch.testing.assert_close(layer.cl, torch.tensor([0.0, 1.0, 10.0]))
        torch.testing.assert_close(layer.cutpoint_features, torch.tensor([0, 0, 1]))
        self.assertIn('base_cutpoints', dict(layer.named_buffers()))
        self.assertIn('cutpoint_radii', dict(layer.named_buffers()))
        self.assertIn('cutpoint_features', dict(layer.named_buffers()))
        self.assertNotIn('cutpoint_deltas', dict(layer.named_parameters()))

    def test_builds_local_movement_radii_for_each_feature(self) -> None:
        layer = BinarizeLayer(
            3,
            (0, 4),
            cutpoints=[
                np.array([]),
                np.array([5.0]),
                np.array([4.0, 0.0]),
                np.array([6.0, 0.0, 2.0]),
            ],
            cutpoint_tuning_eta=0.5,
        )

        torch.testing.assert_close(
            layer.base_cutpoints,
            torch.tensor([5.0, 0.0, 4.0, 0.0, 2.0, 6.0]),
        )
        torch.testing.assert_close(
            layer.cutpoint_radii,
            torch.tensor([0.0, 1.0, 1.0, 0.5, 0.5, 1.0]),
        )
        torch.testing.assert_close(
            layer.cutpoint_features,
            torch.tensor([1, 2, 2, 3, 3, 3]),
        )
        torch.testing.assert_close(layer.cl, layer.base_cutpoints)

    def test_optimizer_updates_cutpoints_within_their_radii(self) -> None:
        layer = BinarizeLayer(
            3,
            (0, 1),
            cutpoints=[np.array([0.0, 2.0, 5.0])],
        )
        optimizer = torch.optim.Adam(layer.parameters(), lr=1.0)
        values = torch.tensor([[1.0]])
        mask = torch.ones_like(values)
        initial_cutpoints = layer.cl.detach().clone()

        output, _output_mask = layer(values, mask)
        output[:, 0].sum().backward()

        self.assertIsNotNone(layer.cutpoint_deltas.grad)
        self.assertNotEqual(layer.cutpoint_deltas.grad[0].item(), 0.0)
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

    def test_only_single_cutpoints_do_not_create_a_parameter(self) -> None:
        layer = BinarizeLayer(
            1,
            (0, 2),
            cutpoints=[np.array([1.0]), np.array([2.0])],
        )

        torch.testing.assert_close(layer.cutpoint_radii, torch.zeros(2))
        self.assertNotIn('cutpoint_deltas', dict(layer.named_parameters()))

    def test_random_mode_still_creates_k_cutpoints_per_feature(self) -> None:
        with patch(
            'torch.randn',
            side_effect=[
                torch.tensor([2.0, 0.0, 1.0]),
                torch.tensor([12.0, 10.0, 11.0]),
            ],
        ):
            layer = BinarizeLayer(3, (0, 2), cutpoint_tuning_eta=0.0)

        torch.testing.assert_close(
            layer.cl,
            torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        )
        torch.testing.assert_close(
            layer.cutpoint_features,
            torch.tensor([0, 0, 0, 1, 1, 1]),
        )

    def test_net_passes_variable_cutpoints_to_the_binarization_layer(self) -> None:
        cutpoints = [np.array([]), np.array([10.0])]

        net = Net(
            [(0, 2), 3, 2],
            use_skip=False,
            cutpoints=cutpoints,
            cutpoint_tuning_eta=0.0,
        )

        torch.testing.assert_close(net.layer_list[0].cl, torch.tensor([10.0]))
        torch.testing.assert_close(
            net.layer_list[0].cutpoint_features,
            torch.tensor([1]),
        )
        self.assertEqual(net.layer_list[1].input_dim, 2)

    def test_net_state_dict_restores_tuned_cutpoints(self) -> None:
        cutpoints = [np.array([0.0, 2.0, 5.0])]
        net = Net(
            [(0, 1), 3, 2],
            use_skip=False,
            cutpoints=cutpoints,
        )
        with torch.no_grad():
            net.layer_list[0].cutpoint_deltas.copy_(torch.tensor([-1.0, 0.5, 1.0]))
        expected = net.layer_list[0].cl.detach().clone()
        state_dict = net.state_dict()
        restored = Net(
            [(0, 1), 3, 2],
            use_skip=False,
            cutpoints=cutpoints,
        )

        restored.load_state_dict(state_dict)

        torch.testing.assert_close(restored.layer_list[0].cl, expected)
        self.assertIn('binary1.base_cutpoints', state_dict)
        self.assertIn('binary1.cutpoint_radii', state_dict)
        self.assertIn('binary1.cutpoint_features', state_dict)
        self.assertIn('binary1.cutpoint_deltas', state_dict)

    def test_bound_names_follow_the_literal_order_and_tuned_values(self) -> None:
        layer = BinarizeLayer(
            2,
            (1, 2),
            cutpoints=[np.array([0.0, 2.0]), np.array([10.0])],
        )
        with torch.no_grad():
            layer.cutpoint_deltas.copy_(torch.tensor([1.0, -1.0, 0.0]))

        layer.get_bound_name(['d', 'a', 'b'], {})

        cutpoints = layer.cl.detach().cpu().numpy()
        self.assertEqual(layer.rule_name[0], 'd')
        self.assertEqual(
            layer.rule_name[1:],
            [
                f'a > {cutpoints[0]}',
                f'a > {cutpoints[1]}',
                f'b > {cutpoints[2]}',
                f'a <= {cutpoints[0]}',
                f'a <= {cutpoints[1]}',
                f'b <= {cutpoints[2]}',
            ],
        )

    def test_rule_extraction_merges_only_bounds_from_the_same_feature(self) -> None:
        previous = BinarizeLayer(
            2,
            (0, 2),
            cutpoints=[np.array([0.0, 1.0]), np.array([10.0])],
            cutpoint_tuning_eta=0.0,
        )
        layer = ConjunctionLayer(3, previous.output_dim)
        with torch.no_grad():
            layer.W.zero_()
            layer.W[[0, 1], 0] = 1.0
            layer.W[[1, 2], 1] = 1.0
            layer.W[[3, 4], 2] = 1.0
        layer.node_activation_cnt = torch.ones(3)
        layer.forward_tot = 2

        _dim2id, rules = extract_rules(previous, None, layer)

        self.assertEqual(
            rules,
            [
                ((-1, 1),),
                ((-1, 1), (-1, 2)),
                ((-1, 3),),
            ],
        )

    def test_disjunction_keeps_the_weakest_bound(self) -> None:
        previous = BinarizeLayer(
            2,
            (0, 2),
            cutpoints=[np.array([0.0, 1.0]), np.array([10.0])],
            cutpoint_tuning_eta=0.0,
        )
        layer = DisjunctionLayer(2, previous.output_dim)
        with torch.no_grad():
            layer.W.zero_()
            layer.W[[0, 1], 0] = 1.0
            layer.W[[3, 4], 1] = 1.0
        layer.node_activation_cnt = torch.ones(2)
        layer.forward_tot = 2

        _dim2id, rules = extract_rules(previous, None, layer)

        self.assertEqual(rules, [((-1, 0),), ((-1, 4),)])

    def test_rejects_mismatched_cutpoint_features(self) -> None:
        with self.assertRaisesRegex(ValueError, 'one array per feature'):
            BinarizeLayer(3, (0, 2), cutpoints=[np.array([1.0])])

    def test_rejects_an_invalid_eta(self) -> None:
        for eta in (-0.1, 1.0):
            with (
                self.subTest(eta=eta),
                self.assertRaisesRegex(ValueError, 'cutpoint_tuning_eta'),
            ):
                BinarizeLayer(3, (0, 1), cutpoint_tuning_eta=eta)
