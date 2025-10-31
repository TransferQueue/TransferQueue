import unittest
# from ..clients.factory import StorageClientFactory
import torch
from transfer_queue.metadata import (
    BatchMeta,
    FieldMeta,
    SampleMeta,
)

from tensordict import TensorDict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.yuanrong_manager import YuanrongStorageManager


class Test(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "client_name": "Yuanrong",
            "host": "127.0.0.1",
            "port": 31501,
            "device_id": 0
        }
        # metadata
        self.field_names = ["text", "label", "mask"]
        self.global_indexes = [8, 9, 10]

        # data: TensorDict
        self.data = TensorDict({
            "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),  # shape: [3, 2]
            "label": torch.tensor([0, 1, 2]),  # shape: [3]
            "mask": torch.tensor([[1], [1], [0]]),  # shape: [3, 1]
        }, batch_size=3)
        samples = []

        for sample_id in range(self.data.batch_size[0]):
            fields_dict = {}
            for field_name in self.data.keys():
                tensor = self.data[field_name][sample_id]
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=tensor.dtype,
                    shape=tensor.shape,
                    production_status=1

                )
                fields_dict[field_name] = field_meta
            sample = SampleMeta(
                global_step=0,
                global_index=self.global_indexes[sample_id],
                fields=fields_dict,
            )
            samples.append(sample)
        self.metadata = BatchMeta(samples=samples)

    # def test_create(self):
    #     self.sm = YuanrongStorageManager(self.cfg)

    def test_generate_keys(self):
        """测试 _generate_yr_keys 生成正确的 key 列表"""
        keys = KVStorageManager._generate_yr_keys(self.metadata)
        expected = [
            '8@label', '9@label', '10@label',
            '8@mask', '9@mask', '10@mask',
            '8@text', '9@text', '10@text'
        ]
        self.assertEqual(keys, expected)
        self.assertEqual(len(keys), 9)  # 3 fields * 3 indexes

    def test_generate_values(self):
        """测试 _generate_values 按 field-major 扁平化 tensor"""
        values = KVStorageManager._generate_values(self.data)
        expected_length = len(self.field_names) * len(self.global_indexes)  # 9
        self.assertEqual(len(values), expected_length)

    def test_generate_values_type_check(self):
        """测试 _generate_values 对非 tensor 输入抛出异常"""
        bad_data = TensorDict({
            "text": torch.tensor([1, 2]),
            "label": "not_a_tensor"
        }, batch_size=2)

        with self.assertRaises(TypeError):
            KVStorageManager._generate_values(bad_data)

    def test_merge_kv_to_tensordict(self):
        """测试 _merge_kv_to_tensordict 能正确重建 TensorDict"""
        # 先生成 values
        values = KVStorageManager._generate_values(self.data)

        # 合并回 TensorDict
        reconstructed = KVStorageManager._merge_kv_to_tensordict(self.metadata, values)

        # print(reconstructed)

        # 检查字段
        self.assertIn("text", reconstructed)
        self.assertIn("label", reconstructed)
        self.assertIn("mask", reconstructed)

        # 检查值是否一致
        self.assertTrue(torch.equal(reconstructed["text"], self.data["text"]))
        self.assertTrue(torch.equal(reconstructed["label"], self.data["label"]))
        self.assertTrue(torch.equal(reconstructed["mask"], self.data["mask"]))

        # 检查 batch_size
        self.assertEqual(reconstructed.batch_size, torch.Size([3]))


if __name__ == "__main__":
    unittest.main()