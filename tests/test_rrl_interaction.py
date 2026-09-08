from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch
from cyclopts import App

from iatreion.configs.model_rrl import RrlBinarization, RrlConfig
from iatreion.rrl.binarization import (
    INTERACTION_BACKGROUNDS,
    INTERACTION_GRID_SIZE,
    MAX_INTERACTION_PAIRS,
    MAX_SAMPLE_SIZE,
    _attention_pairs,
    _interaction_grids,
    _joint_boundary_candidates,
    _sample_indices,
    _select_joint_cutpoints,
    tabpfn_interaction_cutpoints,
)
from iatreion.rrl.experiment import _get_cutpoints
from iatreion.rrl.rrl.models import RRL


class InteractionGridTest(TestCase):
    def test_uses_observed_quantiles_and_all_low_cardinality_values(self) -> None:
        observed = np.r_[np.zeros(80), np.arange(1, 19), np.nan, np.inf]
        X = np.column_stack(
            [observed, np.arange(100) % 3, np.ones(100), np.full(100, np.nan)]
        )

        grids = _interaction_grids(X)

        self.assertLessEqual(len(grids[0]), INTERACTION_GRID_SIZE)
        self.assertTrue(np.all(np.diff(grids[0]) > 0))
        np.testing.assert_array_equal(grids[0][[0, -1]], [0, 18])
        self.assertTrue(np.isin(grids[0], observed).all())
        np.testing.assert_array_equal(grids[0], [0, 5, 12, 18])
        np.testing.assert_array_equal(grids[1], [0, 1, 2])
        np.testing.assert_array_equal(grids[2], [1])
        self.assertEqual(len(grids[3]), 0)

    def test_background_sampling_is_stratified_and_repeatable(self) -> None:
        y = np.repeat([0, 1], 20)

        indices = _sample_indices(y, 7, max_samples=INTERACTION_BACKGROUNDS)

        self.assertEqual(len(np.unique(indices)), INTERACTION_BACKGROUNDS)
        np.testing.assert_array_equal(np.bincount(y[indices]), [4, 4])
        np.testing.assert_array_equal(
            indices, _sample_indices(y, 7, max_samples=INTERACTION_BACKGROUNDS)
        )
        np.testing.assert_array_equal(
            _sample_indices(y[:4], 7, max_samples=INTERACTION_BACKGROUNDS),
            np.arange(4),
        )


class AttentionPairTest(TestCase):
    def test_includes_negative_correlation_and_excludes_ineligible_features(
        self,
    ) -> None:
        first = np.array([-1, -1, 1, 1])
        orthogonal = np.array([-1, 1, -1, 1])
        attention = 0.5 + 0.25 * np.column_stack(
            [first, -first, np.zeros(4), orthogonal, first]
        )
        grids = [np.array([0.0, 1.0]) for _ in range(4)] + [np.array([0.0])]

        self.assertEqual(_attention_pairs(attention, grids), [(0, 1)])

    def test_limits_pairs_and_breaks_ties_by_feature_indices(self) -> None:
        attention = np.tile([[0.25], [0.75]], (1, 8))
        grids = [np.array([0.0, 1.0]) for _ in range(8)]

        pairs = _attention_pairs(attention, grids)

        self.assertEqual(pairs, list(combinations(range(8), 2))[:MAX_INTERACTION_PAIRS])

    def test_constant_attention_has_no_pairs(self) -> None:
        self.assertEqual(
            _attention_pairs(np.full((6, 2), 0.1), [np.arange(2.0), np.arange(2.0)]), []
        )


class JointBoundaryTest(TestCase):
    def test_finds_and_xor_and_multiclass_interactions(self) -> None:
        for name, positive, expected in (
            ('and', [[0, 0], [0, 1]], 2.0),
            ('xor', [[0, 1], [1, 0]], 4.0),
            ('single_feature', [[0, 0], [1, 1]], 0.0),
            ('additive', [[0, 0.25], [0.5, 0.75]], 0.0),
            ('flat', [[0.5, 0.5], [0.5, 0.5]], 0.0),
            ('numerical_noise', [[0, 0], [0, 1e-8]], 0.0),
        ):
            with self.subTest(name=name):
                p = np.asarray(positive)
                classifier = Mock()
                classifier.predict_proba.side_effect = lambda rows, p=p: (
                    np.column_stack(
                        [
                            1 - p[rows[:, 0].astype(int), rows[:, 1].astype(int)],
                            p[rows[:, 0].astype(int), rows[:, 1].astype(int)],
                        ]
                    )
                )
                grids = [np.arange(2.0), np.arange(2.0)]

                candidates = _joint_boundary_candidates(
                    classifier, np.zeros((1, 2)), grids, [(0, 1)], continuous_start=0
                )
                cutpoints = _select_joint_cutpoints(candidates, grids, budget=2)

                if expected:
                    self.assertEqual(candidates, [(expected, 0, 1, 0, 0)])
                    for values in cutpoints:
                        np.testing.assert_array_equal(values, [0.5])
                else:
                    self.assertEqual(candidates, [])
                    self.assertEqual([len(values) for values in cutpoints], [0, 0])

        classifier.predict_proba.return_value = np.array(
            [[0.2, 0.8, 0], [0.2, 0, 0.8], [0.2, 0, 0.8], [0.2, 0.8, 0]]
        )
        classifier.predict_proba.side_effect = None
        candidates = _joint_boundary_candidates(
            classifier, np.zeros((1, 2)), grids, [(0, 1)], continuous_start=0
        )
        self.assertAlmostEqual(candidates[0][0], 3.2)

    def test_opposite_backgrounds_do_not_cancel_and_keep_other_values(self) -> None:
        backgrounds = np.array([[0, 100, 200, np.nan], [1, 101, 201, 17]])
        original = backgrounds.copy()

        def predict(rows):
            positive = (rows[:, 0] > 0) ^ (rows[:, 1] > 1) ^ (rows[:, 2] > 1)
            return np.column_stack([~positive, positive]).astype(float)

        classifier = Mock()
        classifier.predict_proba.side_effect = predict
        grids = [np.array([0.0, 2.0]), np.array([0.0, 2.0]), np.array([17.0])]

        candidates = _joint_boundary_candidates(
            classifier, backgrounds, grids, [(0, 1)], continuous_start=1
        )

        self.assertEqual(candidates, [(4.0, 0, 1, 0, 0)])
        probes = classifier.predict_proba.call_args.args[0]
        np.testing.assert_array_equal(
            probes[:, [0, 3]], np.repeat(backgrounds[:, [0, 3]], 4, axis=0)
        )
        np.testing.assert_array_equal(backgrounds, original)

    def test_batches_large_grids_and_skips_prediction_without_pairs(self) -> None:
        classifier = Mock()
        classifier.predict_proba.side_effect = lambda rows: np.tile(
            [0.5, 0.5], (len(rows), 1)
        )
        backgrounds = np.zeros((INTERACTION_BACKGROUNDS, 2))
        grids = [np.arange(float(INTERACTION_GRID_SIZE)) for _ in range(2)]

        candidates = _joint_boundary_candidates(
            classifier, backgrounds, grids, [(0, 1)], continuous_start=0
        )

        self.assertEqual(candidates, [])
        sizes = [len(call.args[0]) for call in classifier.predict_proba.call_args_list]
        self.assertLessEqual(max(sizes), MAX_SAMPLE_SIZE)
        self.assertEqual(sum(sizes), INTERACTION_BACKGROUNDS * INTERACTION_GRID_SIZE**2)
        classifier.reset_mock()
        self.assertEqual(
            _joint_boundary_candidates(
                classifier, backgrounds, grids, [], continuous_start=0
            ),
            [],
        )
        classifier.predict_proba.assert_not_called()


class JointBudgetTest(TestCase):
    def test_deduplicates_and_skips_pairs_that_do_not_fit(self) -> None:
        grids = [np.array([0.0, 2.0, 4.0]) for _ in range(3)]
        candidates = [
            (10.0, 0, 1, 0, 0),
            (10.0, 0, 1, 0, 0),
            (9.0, 1, 2, 1, 0),
            (8.0, 0, 1, 0, 1),
        ]

        cutpoints = _select_joint_cutpoints(candidates, grids, budget=3)

        np.testing.assert_array_equal(cutpoints[0], [1.0])
        np.testing.assert_array_equal(cutpoints[1], [1.0, 3.0])
        self.assertEqual(len(cutpoints[2]), 0)
        self.assertEqual(sum(map(len, cutpoints)), 3)
        for budget in (0, 1):
            self.assertEqual(
                list(map(len, _select_joint_cutpoints(candidates, grids, budget))),
                [0, 0, 0],
            )

    def test_score_ties_use_feature_then_grid_indices(self) -> None:
        grids = [np.arange(3.0) for _ in range(3)]
        candidates = [(1.0, 1, 2, 0, 0), (1.0, 0, 1, 1, 1), (1.0, 0, 1, 0, 0)]

        cutpoints = _select_joint_cutpoints(candidates, grids, budget=2)

        np.testing.assert_array_equal(cutpoints[0], [0.5])
        np.testing.assert_array_equal(cutpoints[1], [0.5])
        self.assertEqual(len(cutpoints[2]), 0)


class TabpfnInteractionTest(TestCase):
    @patch('iatreion.rrl.binarization.tabpfn_feature_attention')
    @patch('iatreion.rrl.binarization._make_attention_classifier')
    def test_full_fold_teacher_sampled_attention_and_real_backgrounds(
        self, make_classifier: Mock, feature_attention: Mock
    ) -> None:
        rng = np.random.default_rng(7)
        X = np.column_stack(
            [np.arange(300) % 2, rng.choice([-1.0, 1.0], (300, 2)), np.ones(300)]
        )
        y = ((X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(int)

        def attend(_classifier, rows):
            return np.column_stack(
                [
                    np.full(len(rows), 0.1),
                    0.45 + 0.1 * rows[:, 1],
                    0.45 - 0.1 * rows[:, 1],
                    np.zeros(len(rows)),
                ]
            )

        def predict(rows):
            positive = (rows[:, 1] > 0) ^ (rows[:, 2] > 0)
            return np.column_stack([~positive, positive]).astype(float)

        feature_attention.side_effect = attend
        classifier = make_classifier.return_value
        classifier.predict_proba.side_effect = predict
        path = Path('/models/tabpfn-v3.ckpt')

        cutpoints = tabpfn_interaction_cutpoints(
            X, y, continuous_start=1, n_thresholds=1, model_path=path, random_state=7
        )

        make_classifier.assert_called_once_with(path, 7)
        classifier.fit.assert_called_once()
        np.testing.assert_array_equal(classifier.fit.call_args.args[0], X)
        np.testing.assert_array_equal(classifier.fit.call_args.args[1], y)
        np.testing.assert_array_equal(
            feature_attention.call_args.args[1], X[_sample_indices(y, 7)]
        )
        self.assertEqual(len(feature_attention.call_args.args[1]), MAX_SAMPLE_SIZE)
        np.testing.assert_array_equal(cutpoints[0], [0.0])
        np.testing.assert_array_equal(cutpoints[1], [0.0])
        self.assertEqual(len(cutpoints[2]), 0)
        backgrounds = X[_sample_indices(y, 7, max_samples=INTERACTION_BACKGROUNDS)]
        probes = classifier.predict_proba.call_args.args[0]
        np.testing.assert_array_equal(
            probes[:, [0, 3]], np.repeat(backgrounds[:, [0, 3]], 4, axis=0)
        )

        save_model = Mock()
        args = dict(
            dim_list=[(1, 3), 1, 4, 2],
            cutpoints=cutpoints,
            cutpoint_tuning_eta=0.0,
            use_skip=False,
        )
        rrl = RRL(**args, save_model_callback=save_model)
        values = torch.tensor(X, dtype=torch.float32)
        masks = torch.ones_like(values)
        data = [(values, masks, torch.tensor(y))]
        rrl.train_model(
            epoch_advance=Mock(), data_loader=data, valid_loader=data, epoch=1
        )
        restored = RRL(**args)
        restored.net.load_state_dict(save_model.call_args.args[1])
        rrl.net.load_state_dict(save_model.call_args.args[1])
        test_data = [(values, masks)]
        np.testing.assert_allclose(
            restored.predict_proba(test_data), rrl.predict_proba(test_data)
        )

    @patch('iatreion.rrl.binarization._make_attention_classifier')
    def test_insufficient_varying_columns_skip_the_teacher(
        self, make_classifier: Mock
    ) -> None:
        for X, start in (
            (np.ones((4, 2)), 0),
            (np.arange(4.0)[:, None], 0),
            (np.ones((4, 1)), 1),
        ):
            with self.subTest(shape=X.shape, continuous_start=start):
                cutpoints = tabpfn_interaction_cutpoints(
                    X,
                    np.arange(4) % 2,
                    continuous_start=start,
                    n_thresholds=2,
                    model_path=Path('/models/tabpfn-v3.ckpt'),
                    random_state=7,
                )
                self.assertEqual(list(map(len, cutpoints)), [0] * (X.shape[1] - start))
        make_classifier.assert_not_called()

    @patch('iatreion.rrl.binarization.tabpfn_feature_attention')
    @patch('iatreion.rrl.binarization._make_attention_classifier')
    def test_no_correlated_pairs_returns_empty_cutpoints(
        self, make_classifier: Mock, feature_attention: Mock
    ) -> None:
        X = np.arange(8.0).reshape(4, 2)
        feature_attention.return_value = np.ones((4, 2))

        cutpoints = tabpfn_interaction_cutpoints(
            X,
            np.arange(4) % 2,
            continuous_start=0,
            n_thresholds=10,
            model_path=Path('/models/tabpfn-v3.ckpt'),
            random_state=7,
        )

        self.assertEqual(list(map(len, cutpoints)), [0, 0])
        make_classifier.return_value.predict_proba.assert_not_called()

    @patch('iatreion.rrl.experiment.tabpfn_interaction_cutpoints')
    def test_config_parsing_and_experiment_use_only_training_data(
        self, generate: Mock
    ) -> None:
        app = App()

        @app.default
        def parse(*, binarization: RrlBinarization):
            return binarization

        _command, bound, _ignored = app.parse_args(
            ['--binarization', 'tabpfn-interaction']
        )
        mode = bound.arguments['binarization']
        config = RrlConfig(
            dataset=Mock(),
            train=SimpleNamespace(seed=11),
            binarization=mode,
            tabpfn_model_path=Path('/models/tabpfn-v3.ckpt'),
        )
        X = np.arange(30.0).reshape(10, 3)
        y = np.arange(10) % 2
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
        generate.return_value = [np.array([1.0]), np.array([2.0])]

        self.assertIs(_get_cutpoints(config, ctx, 5), generate.return_value)

        np.testing.assert_array_equal(generate.call_args.args[0], X)
        np.testing.assert_array_equal(generate.call_args.args[1], y)
        self.assertEqual(
            generate.call_args.kwargs,
            dict(
                continuous_start=1,
                n_thresholds=5,
                model_path=config.tabpfn_model_path,
                random_state=11,
            ),
        )
        generate.reset_mock()
        config.binarization = 'random'
        self.assertIsNone(_get_cutpoints(config, ctx, 5))
        generate.assert_not_called()
