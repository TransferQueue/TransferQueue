import unittest
#from ..clients.factory import StorageClientFactory
import torch

import sys
import os

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transfer_queue.storage.clients.factory import StorageClientFactory
class Test(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "host": "127.0.0.1",
            "port": 31501,
            "device_id": 0
        }
    def test_create_client(self):
        self.assertIn("Yuanrong", StorageClientFactory._registry)
        #self.assertIs(StorageClientFactory._registry["Yuanrong"], YRStorageClient)
        client=StorageClientFactory.create("Yuanrong", self.cfg)

        with self.assertRaises(ValueError) as cm:
            StorageClientFactory.create("abc",self.cfg)
        self.assertIn("Unknown StorageClient", str(cm.exception))
    def test_client_create_empty_tensorlist(self):
        tensors=[torch.Tensor([2,1]),torch.Tensor([1,5]),torch.Tensor([0]),torch.Tensor([-1.5])]
        shapes=[]
        dtypes=[]
        for t in tensors:
            shapes.append(t.shape)
            dtypes.append(t.dtype)
        client=StorageClientFactory.create("Yuanrong", self.cfg)

        empty_tensors=client._create_empty_tensorlist(shapes,dtypes)
        self.assertEqual(len(tensors),len(empty_tensors))
        for t, et in zip(tensors, empty_tensors):
            self.assertEqual(t.shape,et.shape)
            self.assertEqual(t.dtype,et.dtype)

if __name__ == "__main__":
    unittest.main()