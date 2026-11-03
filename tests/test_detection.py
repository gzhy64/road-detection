"""检查检测匹配、AP 积分、坐标还原与真实数据评估流程。"""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from dataset import RoadSignDataset
from evaluate import average_precision, evaluate_model, match_detections_for_class, summarize_class
from experiment import prepare_splits
from postprocess import nms_by_class, postprocess_single_image


def make_dataset(directory, count=10):
    """生成有真实图片和 XML 文件的小数据集，检查文件读取及坐标变换。"""
    root = Path(directory)
    (root / "images").mkdir()
    (root / "annotations").mkdir()
    for index in range(count):
        Image.new("RGB", (10, 10), (100, 120, 140)).save(root / "images" / f"road{index}.png")
        (root / "annotations" / f"road{index}.xml").write_text(
            '<annotation><object><name>trafficlight</name><bndbox>'
            '<xmin>2</xmin><ymin>2</ymin><xmax>8</xmax><ymax>8</ymax>'
            '</bndbox></object></annotation>', encoding="utf-8")
    return RoadSignDataset(root)


class PerfectDetector(torch.nn.Module):
    """对小数据集给出已知正确预测，隔离模型精度对评估检查的影响。"""
    strides = (8,)

    def forward(self, images):
        batch = images.shape[0]
        logits = images.new_full((batch, 4, 52, 52), -20.0)
        center = images.new_full((batch, 1, 52, 52), -20.0)
        distances = images.new_ones((batch, 4, 52, 52))
        logits[:, 0, 26, 26] = 20.0
        center[:, 0, 26, 26] = 20.0
        # 点 (212,212) 回归到缩放框 [83.2,83.2,332.8,332.8]。
        distances[:, :, 26, 26] = images.new_tensor([128.8, 128.8, 120.8, 120.8])
        return {"cls_logits": [logits], "center_logits": [center], "box_distances": [distances]}


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "boxes": torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.],
                                   [20., 20., 30., 30.], [0., 0., 10., 10.]]),
            "scores": torch.tensor([0.8, 0.95, 0.7, 0.99]),
            "labels": torch.tensor([0, 0, 0, 1]),
        }
        self.target = {"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([0])}

    def test_duplicate_and_wrong_class(self):
        matched = match_detections_for_class(self.result, self.target, 0)
        self.assertEqual(matched["tp"].tolist(), [1., 0., 0.])
        self.assertEqual(matched["fp"].tolist(), [0., 1., 1.])
        other = match_detections_for_class(self.result, self.target, 1)
        self.assertEqual(other["fp"].tolist(), [1.])

    def test_empty_predictions_and_no_gt(self):
        empty = {key: value[:0] for key, value in self.result.items()}
        metrics = summarize_class([match_detections_for_class(empty, self.target, 0)])
        self.assertEqual((metrics["ap50"], metrics["fn"]), (0., 1))
        no_gt = summarize_class([match_detections_for_class(self.result, self.target, 1)])
        self.assertIsNone(no_gt["ap50"])
        self.assertEqual(no_gt["fp"], 1)

    def test_highest_iou_gt_is_not_reused(self):
        target = {"boxes": torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.]]),
                  "labels": torch.tensor([0, 0])}
        matched = match_detections_for_class(self.result, target, 0)
        self.assertEqual(matched["tp"].sum().item(), 1)

    def test_ap_requires_global_confidence_order(self):
        records = [
            {"scores": torch.tensor([0.9]), "tp": torch.tensor([1.]),
             "fp": torch.tensor([0.]), "num_gt": 1},
            {"scores": torch.tensor([0.7, 0.8]), "tp": torch.tensor([1., 0.]),
             "fp": torch.tensor([0., 1.]), "num_gt": 1},
        ]
        metrics = summarize_class(records, score_threshold=0.75)
        self.assertAlmostEqual(metrics["ap50"], 5 / 6, places=6)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertAlmostEqual(average_precision([], []), 0.)


class PipelineTests(unittest.TestCase):
    def test_classwise_nms_keeps_distinct_classes(self):
        boxes = torch.tensor([[0., 0., 10., 10.]]).repeat(3, 1)
        keep = nms_by_class(boxes, torch.tensor([0.8, 0.9, 0.7]), torch.tensor([0, 0, 1]))
        self.assertEqual(keep.tolist(), [1, 2])

    def test_padding_nms_and_inverse_coordinates(self):
        arguments = dict(
            locations=torch.tensor([[100., 100.], [110., 100.], [300., 100.]]),
            cls_logits=torch.tensor([[3., -3., -3., -3.], [2., -3., -3., -3.], [5., -5., -5., -5.]]),
            box_distances=torch.tensor([[40., 20., 60., 50.], [40., 20., 60., 50.], [250., 20., 60., 50.]]),
            center_logits=torch.tensor([2., 2., 5.]),
            resized_size=(416, 278), original_size=(400, 267), score_threshold=0.5,
        )
        result = postprocess_single_image(**arguments)
        torch.testing.assert_close(result["boxes"], torch.tensor([[60 * 267 / 278, 80 * 400 / 416,
                                                                  160 * 267 / 278, 150 * 400 / 416]]))
        self.assertEqual(result["labels"].tolist(), [0])
        arguments["score_threshold"] = 0.999
        self.assertEqual(postprocess_single_image(**arguments)["boxes"].shape, (0, 4))

    def test_real_image_xml_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_dataset(directory)
            model = PerfectDetector().train()
            metrics = evaluate_model(model, dataset, [0, 1, 2], torch.device("cpu"), batch_size=2)
            self.assertAlmostEqual(metrics["map50"], 1.0)
            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (3, 0, 0))
            self.assertTrue(model.training)

    def test_saved_split_and_changed_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_dataset(directory)
            indices, metadata = prepare_splits(dataset, seed=123)
            checkpoint = {"split_seed": 123, "dataset_metadata": metadata}
            restored, restored_metadata = prepare_splits(dataset, checkpoint=checkpoint)
            self.assertEqual(indices, restored)
            self.assertEqual(metadata, restored_metadata)
            annotation = dataset.annotation_paths[0]
            annotation.write_text(annotation.read_text().replace("<xmin>2", "<xmin>1"))
            with self.assertRaises(ValueError):
                prepare_splits(dataset, checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
