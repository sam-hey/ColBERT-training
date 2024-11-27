import os
import torch
import unittest
from colbert.training.utils import manage_checkpoints
from colbert.modeling.colbert import ColBERT
from colbert.infra import ColBERTConfig


class TestManageCheckpoints(unittest.TestCase):
    def setUp(self):
        self.config = ColBERTConfig(
            bsize=32,
            accumsteps=1,
            lr=5e-6,
            maxsteps=500000,
            save_every=1062,
            warmup=1062,
            amp=True,
            checkpoint="bert-base-uncased",
            rank=0,
            nranks=1,
        )
        self.model = ColBERT(name=self.config.checkpoint, colbert_config=self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.checkpoints_path = "test_checkpoints"
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

    # def tearDown(self):
    #     shutil.rmtree(self.checkpoints_path)

    def test_manage_checkpoints(self):
        batch_idx = 2000
        path_save = manage_checkpoints(
            self.config,
            self.model,
            self.optimizer,
            batch_idx,
            savepath=self.checkpoints_path,
        )
        self.assertIsNotNone(path_save)
        self.assertTrue(os.path.exists(path_save))

        config = ColBERTConfig.load_from_checkpoint(path_save)
        self.assertEqual(config.batch_idx, batch_idx)
        self.assertEqual(config.bsize, self.config.bsize)
        self.assertEqual(config.accumsteps, self.config.accumsteps)
        self.assertEqual(config.lr, self.config.lr)
        self.assertEqual(config.maxsteps, self.config.maxsteps)
        self.assertEqual(config.save_every, self.config.save_every)
        self.assertEqual(config.warmup, self.config.warmup)
        self.assertEqual(config.amp, self.config.amp)

        # self.assertIn("model_state_dict", checkpoint)
        # self.assertIn("optimizer_state_dict", checkpoint)
        # self.assertIn("arguments", checkpoint)


if __name__ == "__main__":
    unittest.main()
