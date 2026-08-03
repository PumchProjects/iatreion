import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from iatreion.models.base import Model
from iatreion.models.decision_tree import C45Model, CartModel
from iatreion.models.logistic_regression import LogisticRegressionModel
from iatreion.models.rf import RandomForestModel
from iatreion.models.xgb import XgboostModel
from iatreion.train_utils import TrainStepContext
from iatreion.trainers.model import ModelTrainer


class StubModel(Model):
    def _fit(self, X, y) -> None:
        return None

    def _predict_proba(self, X):
        return np.empty((len(X), 2))


def make_context(feature_names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        is_inner=False,
        name='demo',
        outer_fold=0,
        inner_fold=0,
        db_enc=SimpleNamespace(X_fname=feature_names),
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

    def test_final_lifecycle_exports_native_before_saving(self) -> None:
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.model = Mock()
        trainer._update_config = Mock()
        ctx = make_context(['a'])

        trainer.train_final(ctx)

        self.assertEqual(
            trainer.model.method_calls,
            [
                ('fit', (ctx,), {}),
                ('export_importance', (ctx,), {'methods': {'native'}}),
                ('save_final', (ctx,), {}),
            ],
        )


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
