from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch

from iatreion.configs.model_rrl import RrlConfig
from iatreion.rrl.binarization import (
    MAX_SAMPLE_SIZE,
    _allocate_thresholds,
    _degroup_attention,
    attention_quantile_cutpoints,
    shap_jump_cutpoints,
    tabpfn_attention_cutpoints,
    tabpfn_feature_attention,
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


class RrlCutpointConfigTest(TestCase):
    def test_random_binarization_is_the_default(self) -> None:
        config = RrlConfig(dataset=Mock(), train=Mock())

        self.assertEqual(config.binarization, 'random')
        self.assertIsNone(config.tabpfn_model_path)
        self.assertEqual(config.cutpoint_tuning_eta, 0.5)
        self.assertFalse(hasattr(config, 'trainable_cutpoints'))

    def test_tabpfn_modes_require_a_checkpoint(self) -> None:
        for mode in ('tabpfn-shap', 'tabpfn-attention'):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, 'tabpfn_model_path'),
            ):
                RrlConfig(
                    dataset=Mock(),
                    train=Mock(),
                    binarization=mode,
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


class AttentionAllocationTest(TestCase):
    def test_allocates_proportionally_with_deterministic_ties(self) -> None:
        np.testing.assert_array_equal(
            _allocate_thresholds(
                np.array([3.0, 1.0]),
                np.array([10, 10]),
                4,
            ),
            [3, 1],
        )
        np.testing.assert_array_equal(
            _allocate_thresholds(
                np.ones(2),
                np.array([10, 10]),
                1,
            ),
            [1, 0],
        )

    def test_redistributes_capacity_and_stops_at_total_capacity(self) -> None:
        np.testing.assert_array_equal(
            _allocate_thresholds(
                np.array([3.0, 1.0]),
                np.array([1, 10]),
                4,
            ),
            [1, 3],
        )
        np.testing.assert_array_equal(
            _allocate_thresholds(
                np.array([3.0, 1.0]),
                np.array([1, 1]),
                10,
            ),
            [1, 1],
        )

    def test_degroups_each_token_to_its_three_source_features(self) -> None:
        np.testing.assert_allclose(
            _degroup_attention(np.array([3.0, 0.0, 0.0, 0.0, 0.0])),
            [0.0, 1.0, 1.0, 0.0, 1.0],
        )

    def test_uses_empirical_quantiles_with_duplicates_and_missing_values(
        self,
    ) -> None:
        X = np.column_stack(
            [
                [0.0, 0.0, 0.0, 10.0, 20.0, 30.0, np.nan],
                np.ones(7),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf],
            ]
        )

        cutpoints = attention_quantile_cutpoints(
            X,
            np.array([0.45, 0.1, 0.45]),
            n_thresholds=1,
        )

        np.testing.assert_allclose(cutpoints[0], [5.0, 15.0])
        self.assertEqual(len(cutpoints[1]), 0)
        np.testing.assert_allclose(cutpoints[2], [2.5])
        self.assertTrue(np.all(np.diff(cutpoints[0]) > 0))

    def test_quantile_capacity_reallocates_the_global_budget(self) -> None:
        X = np.column_stack(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                np.arange(6, dtype=float),
            ]
        )

        cutpoints = attention_quantile_cutpoints(
            X,
            np.array([0.9, 0.1]),
            n_thresholds=2,
        )

        self.assertEqual([len(values) for values in cutpoints], [1, 3])
        self.assertTrue(np.all(np.diff(cutpoints[1]) > 0))


class TabpfnFeatureAttentionTest(TestCase):
    class IdentityRope:
        @staticmethod
        def rotate_queries_or_keys(values: torch.Tensor) -> torch.Tensor:
            return values

    def test_aggregates_test_rows_and_excludes_cls_keys(self) -> None:
        attention = SimpleNamespace(
            q_projection=torch.nn.Identity(),
            k_projection=torch.nn.Identity(),
            num_heads=2,
            head_dim=1,
        )
        aggregator = SimpleNamespace(
            blocks=[SimpleNamespace(attention=attention)],
            num_cls_tokens=2,
            rope=self.IdentityRope(),
        )

        class Classifier:
            models_ = [SimpleNamespace(column_aggregator=aggregator)]
            n_features_in_ = 4
            executor_ = SimpleNamespace(
                ensemble_members=[
                    SimpleNamespace(
                        feature_schema=SimpleNamespace(
                            features=[
                                SimpleNamespace(ancestor='f1'),
                                SimpleNamespace(ancestor='f3'),
                            ]
                        )
                    )
                ]
            )

            @staticmethod
            def predict_proba(_X: np.ndarray) -> np.ndarray:
                queries = torch.ones(3, 2, 2)
                keys = torch.tensor(
                    [
                        [
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [20.0, 20.0],
                            [0.0, 0.0],
                        ],
                        [
                            [1000.0, 1000.0],
                            [1000.0, 1000.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [np.log(3), np.log(3)],
                        ],
                    ]
                )
                attention.q_projection(queries)
                attention.k_projection(keys)
                return np.zeros((len(_X), 2))

        scores = tabpfn_feature_attention(Classifier(), np.zeros((2, 4)))

        np.testing.assert_allclose(
            scores,
            [0.0, 11 / 24, 0.0, 13 / 24],
            rtol=1e-6,
        )
        self.assertFalse(attention.q_projection._forward_hooks)
        self.assertFalse(attention.k_projection._forward_hooks)


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
            values=np.zeros((MAX_SAMPLE_SIZE, X.shape[1], 2))
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
        self.assertEqual(X_sample.shape, (MAX_SAMPLE_SIZE, X.shape[1]))
        sampled_y = y[X_sample[:, 0].astype(int)]
        np.testing.assert_array_equal(np.bincount(sampled_y), [128, 128])
        self.assertEqual(explainer_factory.call_args.kwargs['algorithm'], 'permutation')
        self.assertEqual(explainer_factory.call_args.kwargs['seed'], 7)
        np.testing.assert_array_equal(explainer.call_args.args[0], X_sample)
        self.assertEqual(explainer.call_args.kwargs['max_evals'], 7)
        self.assertTrue(explainer.call_args.kwargs['silent'])
        self.assertEqual([len(feature) for feature in cutpoints], [0, 0])


class TabpfnAttentionTest(TestCase):
    @patch('iatreion.rrl.binarization.tabpfn_feature_attention')
    @patch('iatreion.rrl.binarization._make_attention_classifier')
    def test_constant_columns_keep_attention_aligned(
        self,
        make_classifier: Mock,
        feature_attention: Mock,
    ) -> None:
        X = np.column_stack(
            [
                np.zeros(6),
                np.arange(6, dtype=float),
                np.ones(6),
                np.repeat(np.arange(3, dtype=float), 2),
            ]
        )
        y = np.arange(6) % 2
        feature_attention.return_value = np.array([0.0, 0.6, 0.0, 0.4])

        cutpoints = tabpfn_attention_cutpoints(
            X,
            y,
            continuous_start=1,
            n_thresholds=1,
            model_path=Path('/models/tabpfn-v3.ckpt'),
            random_state=7,
        )

        np.testing.assert_allclose(cutpoints[0], [1.5, 3.5])
        self.assertEqual(len(cutpoints[1]), 0)
        np.testing.assert_allclose(cutpoints[2], [1.5])
        np.testing.assert_array_equal(
            feature_attention.call_args.args[1],
            X,
        )

    @patch('iatreion.rrl.binarization._explain_sample')
    @patch('iatreion.rrl.binarization.attention_quantile_cutpoints')
    @patch('iatreion.rrl.binarization.tabpfn_feature_attention')
    @patch('iatreion.rrl.binarization._sample_indices')
    @patch('iatreion.rrl.binarization._make_attention_classifier')
    def test_uses_sampled_attention_and_full_fold_quantiles_without_shap(
        self,
        make_classifier: Mock,
        sample_indices: Mock,
        feature_attention: Mock,
        select_cutpoints: Mock,
        explain_sample: Mock,
    ) -> None:
        X = np.arange(30, dtype=float).reshape(10, 3)
        y = np.arange(10) % 2
        indices = np.array([1, 3, 5])
        X_sample = X[indices]
        sample_indices.return_value = indices
        feature_attention.return_value = np.array([0.1, 0.6, 0.3])
        selected = [np.array([1.0]), np.array([2.0])]
        select_cutpoints.return_value = selected
        path = Path('/models/tabpfn-v3.ckpt')

        result = tabpfn_attention_cutpoints(
            X,
            y,
            continuous_start=1,
            n_thresholds=4,
            model_path=path,
            random_state=7,
        )

        self.assertIs(result, selected)
        make_classifier.assert_called_once_with(path, 7)
        classifier = make_classifier.return_value
        np.testing.assert_array_equal(classifier.fit.call_args.args[0], X)
        np.testing.assert_array_equal(classifier.fit.call_args.args[1], y)
        sample_indices.assert_called_once_with(y, 7)
        self.assertIs(feature_attention.call_args.args[0], classifier)
        np.testing.assert_array_equal(feature_attention.call_args.args[1], X_sample)
        np.testing.assert_array_equal(
            select_cutpoints.call_args.args[0], X[:, 1:]
        )
        np.testing.assert_array_equal(
            select_cutpoints.call_args.args[1], np.array([0.6, 0.3])
        )
        self.assertEqual(select_cutpoints.call_args.kwargs, {'n_thresholds': 4})
        explain_sample.assert_not_called()


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

    @patch('iatreion.rrl.experiment.tabpfn_attention_cutpoints')
    def test_attention_mode_passes_only_training_data(self, generate: Mock) -> None:
        X = np.arange(30, dtype=float).reshape(10, 3)
        y = np.arange(10) % 2
        generated = [np.array([1.0]), np.array([2.0, 3.0])]
        generate.return_value = generated
        path = Path('/models/tabpfn-v3.ckpt')
        args = SimpleNamespace(
            binarization='tabpfn-attention',
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

    @patch('iatreion.rrl.experiment.tabpfn_attention_cutpoints')
    @patch('iatreion.rrl.experiment.tabpfn_shap_cutpoints')
    def test_random_mode_does_not_build_a_teacher(
        self,
        generate_shap: Mock,
        generate_attention: Mock,
    ) -> None:
        args = SimpleNamespace(binarization='random')
        ctx = SimpleNamespace(
            db_enc=SimpleNamespace(categorical_flen=0, numeric_flen=2)
        )

        self.assertIsNone(_get_cutpoints(args, ctx, 5))
        generate_shap.assert_not_called()
        generate_attention.assert_not_called()


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
