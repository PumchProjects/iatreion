import json
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from iatreion.models.base import Model
from iatreion.models.decision_tree import C45Model, CartModel
from iatreion.models.importance import (
    ShapBundle,
    TreeShapWorkerError,
    _run_random_forest_tree_shap,
    calc_random_forest_shap_importance,
    get_importance_sample,
    save_shap_bundle,
)
from iatreion.models.logistic_regression import LogisticRegressionModel
from iatreion.models.rf import RandomForestModel
from iatreion.models.xgb import XgboostModel, _validate_shap_additivity
from iatreion.show_helpers.shap import _load_shap_explanation
from iatreion.train_utils import TrainStepContext
from iatreion.trainers.model import ModelTrainer


class StubModel(Model):
    def _fit(self, X, y) -> None:
        return None

    def _predict_proba(self, X):
        return np.empty((len(X), 2))


def make_context(
    feature_names: list[str],
    *,
    train_data: tuple[np.ndarray, np.ndarray] | None = None,
    test_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimpleNamespace:
    n_features = len(feature_names)
    return SimpleNamespace(
        is_inner=False,
        name='demo',
        outer_fold=0,
        inner_fold=0,
        db_enc=SimpleNamespace(X_fname=feature_names),
        train_data=train_data
        or (np.ones((4, n_features), dtype=float), np.array([0, 1, 0, 1])),
        test_data=test_data
        or (np.ones((2, n_features), dtype=float), np.array([0, 1])),
        shap_file='shap_demo_0_0.npz',
    )


def make_shap_config(root: Path, *, final: bool = False, num_class: int = 2):
    labels = ['negative', 'positive'] if num_class == 2 else ['a', 'b', 'c']
    return SimpleNamespace(
        dataset=SimpleNamespace(_encode=False),
        train=SimpleNamespace(
            final=final,
            seed=42,
            num_class=num_class,
            group_labels=labels,
            _log_dir=root,
        ),
        importance_max_samples=256,
        importance_repeats=1,
    )


class ImportanceLifecycleTest(TestCase):
    @patch('iatreion.models.base.save_importance_score')
    def test_internal_exports_all_requested_methods(self, save: Mock) -> None:
        model = StubModel()
        model.config = SimpleNamespace(
            fold_scope='outer',
            importance_methods=['native', 'permutation', 'shap'],
        )
        model._calc_native_importance = Mock(return_value={'a': 1.0})
        model._calc_permutation_importance = Mock(return_value={'a': 2.0})
        model._calc_shap_importance = Mock(return_value={'a': 3.0})
        ctx = make_context(['a'])

        model.export_importance(ctx)

        model._calc_native_importance.assert_called_once_with(ctx)
        model._calc_permutation_importance.assert_called_once_with(ctx)
        model._calc_shap_importance.assert_called_once_with(ctx)
        self.assertEqual(
            [call.kwargs['method'] for call in save.call_args_list],
            ['native', 'permutation', 'shap'],
        )

    @patch('iatreion.models.base.save_importance_score')
    def test_final_filter_exports_requested_native_only(self, save: Mock) -> None:
        model = StubModel()
        model.config = SimpleNamespace(
            fold_scope='outer',
            importance_methods=['native', 'permutation', 'shap'],
        )
        model._calc_native_importance = Mock(return_value={'a': 1.0})
        model._calc_permutation_importance = Mock()
        model._calc_shap_importance = Mock()
        ctx = make_context(['a'])

        model.export_importance(ctx, methods={'native'})

        model._calc_native_importance.assert_called_once_with(ctx)
        model._calc_permutation_importance.assert_not_called()
        model._calc_shap_importance.assert_not_called()
        save.assert_called_once_with(
            model.config,
            ctx,
            {'a': 1.0},
            method='native',
        )

    @patch('iatreion.models.base.save_importance_score')
    def test_final_filter_does_nothing_when_native_is_not_requested(
        self, save: Mock
    ) -> None:
        model = StubModel()
        model.config = SimpleNamespace(
            fold_scope='outer',
            importance_methods=['shap'],
        )
        model._calc_native_importance = Mock()

        model.export_importance(make_context(['a']), methods={'native'})

        model._calc_native_importance.assert_not_called()
        save.assert_not_called()

    def test_importance_errors_are_not_suppressed(self) -> None:
        model = StubModel()
        model.config = SimpleNamespace(
            fold_scope='outer',
            importance_methods=['native'],
        )
        model._calc_native_importance = Mock(side_effect=RuntimeError('broken'))

        with self.assertRaisesRegex(RuntimeError, 'broken'):
            model.export_importance(make_context(['a']))

    def test_final_lifecycle_saves_before_exporting_requested_methods(self) -> None:
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.model = Mock()
        trainer._update_config = Mock()
        ctx = make_context(['a'])

        trainer.train_final(ctx)

        self.assertEqual(
            trainer.model.method_calls,
            [
                ('fit', (ctx,), {}),
                ('save_final', (ctx,), {}),
                ('export_importance', (ctx,), {}),
            ],
        )

    def test_final_importance_samples_training_data(self) -> None:
        X_train = np.arange(12, dtype=float).reshape(6, 2)
        y_train = np.array([0, 1, 0, 1, 0, 1])
        ctx = make_context(
            ['a', 'b'],
            train_data=(X_train, y_train),
            test_data=(np.empty((0, 2)), np.empty(0, dtype=int)),
        )
        with TemporaryDirectory() as tmp:
            sample = get_importance_sample(
                make_shap_config(Path(tmp), final=True),
                ctx,
            )

        np.testing.assert_array_equal(sample.X, X_train)
        np.testing.assert_array_equal(sample.y, y_train)
        self.assertEqual(sample.data_scope, 'training')


class NativeImportanceTest(TestCase):
    def test_xgboost_maps_feature_indices_and_fills_unused_features(self) -> None:
        model = XgboostModel.__new__(XgboostModel)
        model.bst = Mock()
        model.bst.get_score.return_value = {'f0': 2.5, 'f2': 0.75}

        score = model._calc_native_importance(make_context(['a', 'b', 'c']))

        self.assertEqual(score, {'a': 2.5, 'b': 0.0, 'c': 0.75})
        model.bst.get_score.assert_called_once_with(importance_type='gain')

    def test_random_forest_exports_impurity_importance(self) -> None:
        model = RandomForestModel.__new__(RandomForestModel)
        model.forest = SimpleNamespace(feature_importances_=np.array([0.2, 0.8]))

        score = model._calc_native_importance(make_context(['a', 'b']))

        self.assertEqual(score, {'a': 0.2, 'b': 0.8})

    def test_c45_exports_impurity_importance(self) -> None:
        model = C45Model.__new__(C45Model)
        model.estimator = SimpleNamespace(feature_importances_=np.array([0.3, 0.7]))

        score = model._calc_native_importance(make_context(['a', 'b']))

        self.assertEqual(score, {'a': 0.3, 'b': 0.7})

    def test_cart_exports_impurity_importance(self) -> None:
        model = CartModel.__new__(CartModel)
        model.estimator = SimpleNamespace(feature_importances_=np.array([0.4, 0.6]))

        score = model._calc_native_importance(make_context(['a', 'b']))

        self.assertEqual(score, {'a': 0.4, 'b': 0.6})

    def test_logistic_regression_exports_mean_absolute_coefficients(self) -> None:
        model = LogisticRegressionModel.__new__(LogisticRegressionModel)
        model.estimator = SimpleNamespace(
            coef_=np.array(
                [
                    [-1.0, 2.0],
                    [3.0, -4.0],
                ]
            )
        )

        score = model._calc_native_importance(make_context(['a', 'b']))

        self.assertEqual(score, {'a': 2.0, 'b': 3.0})


class ShapBackendTest(TestCase):
    def test_xgboost_additivity_accepts_float32_rounding_error(self) -> None:
        contributions = np.array(
            [[[0.125, -0.12135589]]],
            dtype=np.float32,
        )
        margins = np.array([[0.00364518]], dtype=np.float32)
        error = abs(float(contributions.sum()) - float(margins.item()))

        self.assertGreater(error, 1e-6)
        self.assertLess(error, 2e-6)

        _validate_shap_additivity(contributions, margins)

    def test_xgboost_additivity_rejects_material_error(self) -> None:
        contributions = np.array(
            [[[0.125, -0.12135482]]],
            dtype=np.float32,
        )
        margins = np.array([[0.00464518]], dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError,
            r'max_abs_error=.*rtol=1e-05, atol=1e-05',
        ):
            _validate_shap_additivity(contributions, margins)

    def test_xgboost_native_contributions_binary(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        booster = xgb.train(
            {'objective': 'binary:logistic', 'device': 'cpu', 'seed': 3},
            xgb.DMatrix(X, y),
            num_boost_round=4,
        )
        with TemporaryDirectory() as tmp:
            model = XgboostModel.__new__(XgboostModel)
            model.config = make_shap_config(Path(tmp))
            model.bst = booster
            model.feature_types = ['q', 'q', 'q']
            model.num_class = 2
            ctx = make_context(['a', 'b', 'c'], test_data=(X, y))

            score = model._calc_shap_importance(ctx)
            with np.load(Path(tmp) / ctx.shap_file) as arrays:
                values = np.asarray(arrays['values'])
                base_values = np.asarray(arrays['base_values'])
                data = np.asarray(arrays['data'])
                explainer = str(arrays['explainer'].item())
                output_space = str(arrays['output_space'].item())

        margins = booster.predict(xgb.DMatrix(data), output_margin=True)
        np.testing.assert_allclose(
            values.sum(axis=1) + base_values,
            margins,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(values.shape, (20, 3))
        self.assertEqual(set(score), {'a', 'b', 'c'})
        self.assertEqual(explainer, 'xgboost-pred-contribs')
        self.assertEqual(output_space, 'raw-margin')

    def test_xgboost_native_contributions_multiclass(self) -> None:
        rng = np.random.default_rng(4)
        X = rng.normal(size=(24, 2))
        y = np.arange(24) % 3
        booster = xgb.train(
            {
                'objective': 'multi:softprob',
                'num_class': 3,
                'device': 'cpu',
                'seed': 4,
            },
            xgb.DMatrix(X, y),
            num_boost_round=3,
        )
        with TemporaryDirectory() as tmp:
            model = XgboostModel.__new__(XgboostModel)
            model.config = make_shap_config(Path(tmp), num_class=3)
            model.bst = booster
            model.feature_types = ['q', 'q']
            model.num_class = 3
            ctx = make_context(['a', 'b'], test_data=(X, y))

            model._calc_shap_importance(ctx)
            with np.load(Path(tmp) / ctx.shap_file) as arrays:
                values = np.asarray(arrays['values'])
                base_values = np.asarray(arrays['base_values'])
                data = np.asarray(arrays['data'])

        margins = booster.predict(
            xgb.DMatrix(data), output_margin=True, strict_shape=True
        )
        np.testing.assert_allclose(
            values.sum(axis=1) + base_values,
            margins,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(values.shape, (24, 2, 3))

    def test_random_forest_path_dependent_worker(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(size=(16, 3))
        y = (X[:, 0] > 0).astype(int)
        forest = RandomForestClassifier(n_estimators=5, random_state=5).fit(X, y)
        with TemporaryDirectory() as tmp:
            config = make_shap_config(Path(tmp))
            ctx = make_context(['a', 'b', 'c'], test_data=(X, y))

            score = calc_random_forest_shap_importance(config, ctx, forest)
            with np.load(Path(tmp) / ctx.shap_file) as arrays:
                values = np.asarray(arrays['values'])
                base_values = np.asarray(arrays['base_values'])
                data = np.asarray(arrays['data'])
                self.assertEqual(
                    str(arrays['explainer'].item()), 'tree-path-dependent'
                )
                self.assertEqual(str(arrays['output_space'].item()), 'probability')

        np.testing.assert_allclose(
            values.sum(axis=1) + base_values,
            forest.predict_proba(data),
            atol=1e-12,
        )
        self.assertEqual(set(score), {'a', 'b', 'c'})

    @patch('iatreion.models.importance.mp.get_context')
    def test_worker_sigsegv_becomes_python_error(self, get_context: Mock) -> None:
        process = Mock(exitcode=-signal.SIGSEGV)
        get_context.return_value.Process.return_value = process

        with self.assertRaisesRegex(TreeShapWorkerError, 'SIGSEGV'):
            _run_random_forest_tree_shap(Mock(), np.ones((2, 1)), ['a'])

        process.start.assert_called_once_with()
        process.join.assert_called_once_with()

    @patch('iatreion.models.rf.calc_shap_importance')
    @patch('iatreion.models.rf.calc_random_forest_shap_importance')
    def test_random_forest_worker_failure_falls_back(
        self,
        tree_shap: Mock,
        permutation_shap: Mock,
    ) -> None:
        tree_shap.side_effect = TreeShapWorkerError('SIGSEGV')
        permutation_shap.return_value = {'a': 1.0}
        model = RandomForestModel.__new__(RandomForestModel)
        model.config = Mock()
        model.forest = Mock()
        ctx = make_context(['a'])

        self.assertEqual(model._calc_shap_importance(ctx), {'a': 1.0})
        permutation_shap.assert_called_once_with(
            model.config,
            ctx,
            model._predict_proba,
            explainer_name='permutation-fallback',
        )


class ImportanceOutputTest(TestCase):
    def test_final_score_name_keeps_existing_format(self) -> None:
        ctx = make_context(['a'])
        ctx.get_importance_file = TrainStepContext.get_importance_file.__get__(
            ctx, TrainStepContext
        )
        with TemporaryDirectory() as tmp:
            from iatreion.models.importance import save_importance_score

            config = SimpleNamespace(
                dataset=SimpleNamespace(_encode=False),
                train=SimpleNamespace(_log_dir=Path(tmp)),
            )
            save_importance_score(config, ctx, {'a': 1.0}, method='native')
            score_file = Path(tmp) / 'score_native_demo_0_0.json'
            score = json.loads(score_file.read_text())

        self.assertEqual(score, {'a': 1.0})

    def test_shap_bundle_writes_metadata(self) -> None:
        ctx = make_context(['a'])
        bundle = ShapBundle(
            values=np.ones((2, 1)),
            base_values=np.zeros(2),
            data=np.ones((2, 1)),
            y_true=np.array([0, 1]),
            sample_indices=np.array([0, 1]),
            feature_names=['a'],
            output_names=['positive'],
            explainer='test-explainer',
            output_space='probability',
            data_scope='training',
        )
        with TemporaryDirectory() as tmp:
            train = SimpleNamespace(_log_dir=Path(tmp))
            save_shap_bundle(train, ctx, bundle)
            loaded = _load_shap_explanation(Path(tmp) / ctx.shap_file)

        self.assertEqual(loaded.explainer, 'test-explainer')
        self.assertEqual(loaded.output_space, 'probability')
        self.assertEqual(loaded.data_scope, 'training')

    def test_legacy_shap_bundle_metadata_defaults_to_unknown(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.npz'
            np.savez_compressed(
                path,
                values=np.ones((2, 1)),
                base_values=np.zeros(2),
                data=np.ones((2, 1)),
                y_true=np.array([0, 1]),
                sample_indices=np.array([0, 1]),
                feature_names=np.array(['a']),
                output_names=np.array(['positive']),
            )
            loaded = _load_shap_explanation(path)

        self.assertEqual(loaded.explainer, 'unknown')
        self.assertEqual(loaded.output_space, 'unknown')
        self.assertEqual(loaded.data_scope, 'unknown')
