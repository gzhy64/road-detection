"""检查完整反向传播、学习率边界和检查点恢复后的更新一致性。"""

import tempfile
import unittest
from pathlib import Path

import torch

from experiment import load_checkpoint
from model import FCOS
from train import EpochLRScheduler, restore_random_state, save_checkpoint, train_one_batch


class TrainingTests(unittest.TestCase):
    def test_full_model_backward_and_empty_target(self):
        torch.manual_seed(7)
        model = FCOS()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        images = torch.rand(1, 3, 128, 128)
        targets = [{"boxes": torch.tensor([[32., 32., 96., 96.]]),
                    "labels": torch.tensor([0]), "resized_size": (128, 128)}]
        before = next(model.backbone.parameters()).detach().clone()
        values = train_one_batch(model, images, targets, optimizer, torch.device("cpu"))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in values.values()))
        self.assertFalse(torch.equal(before, next(model.backbone.parameters())))
        targets[0]["boxes"] = torch.empty(0, 4)
        targets[0]["labels"] = torch.empty(0, dtype=torch.int64)
        values = train_one_batch(model, images, targets, optimizer, torch.device("cpu"))
        self.assertEqual(values["loss_box"], 0.)
        self.assertEqual(values["loss_center"], 0.)

    def test_lr_boundaries_and_legacy_epoch(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = EpochLRScheduler(optimizer, (2, 3), 0.1)
        rates = []
        for _ in range(4):
            rates.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        torch.testing.assert_close(torch.tensor(rates), torch.tensor([1e-3, 1e-3, 1e-4, 1e-5]))
        legacy_optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        EpochLRScheduler(legacy_optimizer, (30, 40), 0.1, completed_epochs=35)
        self.assertAlmostEqual(legacy_optimizer.param_groups[0]["lr"], 1e-4)

    def test_resume_preserves_next_update(self):
        torch.manual_seed(9)
        model = torch.nn.Linear(2, 1)
        model.num_classes = 4
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = EpochLRScheduler(optimizer, (1, 3), 0.1)
        generator = torch.Generator().manual_seed(2026)

        def update(network, optim, gen):
            order = torch.randperm(5, generator=gen)
            x = torch.randn(5, 2)[order]
            optim.zero_grad()
            network(x).square().sum().backward()
            optim.step()
            return order

        update(model, optimizer, generator)
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pth"
            save_checkpoint(model, optimizer, 1, {}, {}, path, 2026,
                            scheduler=scheduler, train_generator=generator)
            expected_order = update(model, optimizer, generator)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            checkpoint = load_checkpoint(path)
            restored = torch.nn.Linear(2, 1)
            restored.load_state_dict(checkpoint["model_state_dict"])
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
            restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            restored_scheduler = EpochLRScheduler(restored_optimizer)
            restored_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            restored_generator = torch.Generator()
            restore_random_state(checkpoint, restored_generator)
            actual_order = update(restored, restored_optimizer, restored_generator)
            self.assertTrue(torch.equal(expected_order, actual_order))
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])


if __name__ == "__main__":
    unittest.main()
