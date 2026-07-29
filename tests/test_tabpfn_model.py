from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from iatreion.models.tabpfn import TABPFN_MODEL_FILE, TabPFNModel
from iatreion.train_utils.artifacts import (
    get_artifact_dir,
    get_transform_artifact_path,
)


class TabPFNModelTest(TestCase):
    @patch('iatreion.models.tabpfn.TabPFNClassifier')
    def test_uses_explicit_checkpoint(self, classifier: Mock) -> None:
        checkpoint = Path('/models/tabpfn-v3.ckpt')
        config = SimpleNamespace(model_path=checkpoint, n_jobs=3)

        TabPFNModel(config)

        classifier.assert_called_once_with(
            model_path=str(checkpoint),
            memory_saving_mode=False,
            random_state=0,
            n_preprocessing_jobs=3,
        )

    def test_saves_final_model_and_transform(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TabPFNModel.__new__(TabPFNModel)
            model.config = SimpleNamespace(train=SimpleNamespace(_log_dir=root))
            model.model = Mock()
            ctx = SimpleNamespace(name='h-demo', db_enc=Mock())

            model.save_final(ctx)

            artifact_dir = get_artifact_dir(root, ctx.name)
            ctx.db_enc.save_transform_artifact.assert_called_once_with(
                get_transform_artifact_path(root, ctx.name)
            )
            model.model.save_fit_state.assert_called_once_with(
                artifact_dir / TABPFN_MODEL_FILE
            )

    @patch('iatreion.models.tabpfn.TabPFNClassifier')
    def test_loads_final_model_on_automatic_device(self, classifier: Mock) -> None:
        restored = Mock()
        classifier.load_from_fit_state.return_value = restored
        model = TabPFNModel.__new__(TabPFNModel)
        model.model = Mock()
        artifact_dir = Path('/artifacts/h-demo')

        model.load_final(artifact_dir, Mock())

        classifier.load_from_fit_state.assert_called_once_with(
            artifact_dir / TABPFN_MODEL_FILE,
            device='auto',
        )
        self.assertIs(model.model, restored)
