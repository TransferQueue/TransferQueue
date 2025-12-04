import time

import pytest
import ray
import torch

from transfer_queue.storage.clients.ray_storage_client import RayStorageClient


@pytest.fixture(scope="session")
def ray_setup():
    ray.init(address="auto", ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def ray_storage_client(ray_setup):
    client = RayStorageClient.options(resources={"node:dev116": 1}).remote()

    def get_node_id():
        return ray.get_runtime_context().get_node_id()

    actor_node = ray.get(client.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id()))
    driver_node = ray.get(get_node_id.remote())

    print(f"Driver running on: {driver_node}")
    print(f"RayStorageClient actor running on: {actor_node}")
    assert actor_node != driver_node, "Acotr must run on different nodes!"
    yield client
    client.clear(["tensor_0", "tensor_1", "tensor_2", "gpu_tensor_0", "gpu_tensor_1"])


def test_ray_storage_put_get(ray_storage_client):
    keys = ["tensor_0", "tensor_1", "tensor_2"]
    values = [torch.randn(10, 20), torch.randn(5, 15), torch.randn(8, 12)]

    ray_storage_client.put(keys, values)

    retrieved = ray_storage_client.get(keys)

    for original, retrieved_tensor in zip(values, retrieved, strict=False):
        torch.testing.assert_close(original, retrieved_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ray_storage_put_get_gpu(ray_storage_client):
    keys = ["gpu_tensor_0", "gpu_tensor_1"]
    values = [torch.randn(100, 200).cuda(), torch.randn(50, 150).cuda()]

    assert all(v.is_cuda for v in values)

    ray.get(ray_storage_client.put.remote(keys, values))
    retrieved = ray.get(ray_storage_client.get.remote(keys))

    for retrieved_tensor in retrieved:
        assert retrieved_tensor.device.type == "cuda"
        print(f"Tensor on device: {retrieved_tensor.device}")


def test_nixl_vs_object_store_performance(ray_storage_client):
    large_tensor = torch.randn(10000, 10000).cuda()

    print("Warming up NIXL...")
    warmup_keys = ["warmup"]

    ray.get(ray_storage_client.set_use_gpu.remote(True))
    ray.get(ray_storage_client.put.remote(warmup_keys, [large_tensor]))
    ray.get(ray_storage_client.get.remote(warmup_keys))
    ray.get(ray_storage_client.clear.remote(warmup_keys))

    print("=== Test: NIXL (use_gpu=True) ===")
    ray.get(ray_storage_client.set_use_gpu.remote(True))
    start = time.time()
    ray.get(ray_storage_client.put.remote(["test"], [large_tensor]))
    ray.get(ray_storage_client.get.remote(["test"]))
    nixl_time = time.time() - start
    print(f"====== transfer with NIXL cost {nixl_time:.4f}s ======")
    ray.get(ray_storage_client.clear.remote(["test"]))

    print("\n=== Test: Object Store (use_gpu=False) ===")
    ray.get(ray_storage_client.set_use_gpu.remote(False))
    start = time.time()
    ray.get(ray_storage_client.put.remote(["test"], [large_tensor]))
    ray.get(ray_storage_client.get.remote(["test"]))
    obj_time = time.time() - start
    print(f"====== transfer with object_store cost {obj_time:.4f}s ======")
    ray.get(ray_storage_client.clear.remote(["test"]))


# -----------------------------
# config test parameters
# -----------------------------
NUM_WARMUP = 1
NUM_REPEAT = 1

TEST_CONFIGS: list[tuple[tuple[int, int], torch.dtype]] = [
    ((10000, 10000), torch.float32),
    ((20000, 20000), torch.float32),
    ((30000, 30000), torch.float32),
    ((40000, 40000), torch.float32),
    ((10000, 10000), torch.float16),
    ((20000, 20000), torch.float16),
    ((30000, 30000), torch.float16),
    ((40000, 40000), torch.float16),
    ((10000, 10000), torch.float64),
    ((20000, 20000), torch.float64),
    ((30000, 30000), torch.float64),
    ((5000, 5000), torch.float32),
]


@pytest.mark.parametrize("tensor_shape,dtype", TEST_CONFIGS)
def test_nixl_vs_object_store_performance_multi_tensors(
    ray_storage_client, tensor_shape: tuple[int, int], dtype: torch.dtype
):
    device = "cpu"

    large_tensor = torch.randn(*tensor_shape, dtype=dtype).to(device)

    element_size_bytes = large_tensor.element_size()
    tensor_size_mb = (large_tensor.numel() * element_size_bytes) / (1024**2)
    print(f"Testing tensor size: {tensor_shape}, dtype: {dtype}, total memory ≈ {tensor_size_mb:.2f} MB")

    # warmup NIXL
    warmup_key = "warmup"
    ray.get(ray_storage_client.set_use_gpu.remote(True))
    ray.get(ray_storage_client.put.remote([warmup_key], [large_tensor]))
    ray.get(ray_storage_client.get.remote([warmup_key]))
    ray.get(ray_storage_client.clear.remote([warmup_key]))

    # test NIXL（use_gpu=True）
    print("=== Test: NIXL (GPU Direct) ===")
    ray.get(ray_storage_client.set_use_gpu.remote(True))
    start = time.time()
    ray.get(ray_storage_client.put.remote(["test_nixl"], [large_tensor]))
    result_tensors = ray.get(ray_storage_client.get.remote(["test_nixl"]))
    nixl_time = time.time() - start
    print(f"NIXL transfer time: {nixl_time:.4f}s")
    ray.get(ray_storage_client.clear.remote(["test_nixl"]))

    assert len(result_tensors) == 1
    assert result_tensors[0].shape == large_tensor.shape
    assert result_tensors[0].dtype == large_tensor.dtype

    # test Object Store（use_gpu=False）
    print("=== Test: Object Store (CPU fallback) ===")
    ray.get(ray_storage_client.set_use_gpu.remote(False))
    start = time.time()
    ray.get(ray_storage_client.put.remote(["test_obj"], [large_tensor]))
    result_tensors = ray.get(ray_storage_client.get.remote(["test_obj"]))
    obj_time = time.time() - start
    print(f"Object Store transfer time: {obj_time:.4f}s")
    ray.get(ray_storage_client.clear.remote(["test_obj"]))

    assert len(result_tensors) == 1
    assert result_tensors[0].shape == large_tensor.shape
    assert result_tensors[0].dtype == large_tensor.dtype

    speedup = obj_time / nixl_time if nixl_time > 0 else float("inf")
    latency_improvement = (obj_time - nixl_time) / obj_time * 100 if nixl_time > 0 else float("inf")
    print(f"Result: tensor size: {tensor_shape}, dtype: {dtype}, total memory ≈ {tensor_size_mb:.2f} MB")
    print(
        f"NIXL Time: {nixl_time:.4f}s | Object Store Time: {obj_time:.4f}s | "
        f"Speedup: {speedup:.2f}x | latency_improvement: {latency_improvement:.2f}%"
    )

    # Multi-client concurrent testing


def test_multiple_clients_concurrent(ray_setup):
    num_clients = 3
    clients = [RayStorageClient({"device_id": 0}) for _ in range(num_clients)]

    for i, client in enumerate(clients):
        keys = [f"client_{i}_tensor_{j}" for j in range(3)]
        values = [torch.randn(10, 10) * j for j in range(3)]
        client.put(keys, values)

    for i, client in enumerate(clients):
        keys = [f"client_{i}_tensor_{j}" for j in range(3)]
        shapes = [(10, 10)] * 3
        dtypes = [torch.float32] * 3
        retrieved = client.get(keys, shapes=shapes, dtypes=dtypes)

        for tensor in retrieved:
            assert not tensor.is_cuda
            assert tensor.shape == (10, 10)
