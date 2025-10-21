<div align="center">
  <h2 align="center">
      TransferQueue: An asynchronous streaming data management module for efficient post-training
  </h2>
  <a href="https://arxiv.org/abs/2507.01663" target="_blank"><strong>Paper</strong></a>
| <a href="https://zhuanlan.zhihu.com/p/1930244241625449814" target="_blank"><strong>Zhihu</strong></a>
  <br />
  <br />

  <a href="https://deepwiki.com/TransferQueue/TransferQueue"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
  [![GitHub Repo stars](https://img.shields.io/github/stars/TransferQueue/TransferQueue)](https://github.com/TransferQueue/TransferQueue/stargazers/)
  [![GitHub commit activity](https://img.shields.io/github/commit-activity/w/TransferQueue/TransferQueue)](https://github.com/TransferQueue/TransferQueue/graphs/commit-activity)

</div>
<br/>



<h2 id="overview">🎉 Overview</h2>

TransferQueue is a high-performance data storage and transfer module with panoramic data visibility and streaming scheduling capabilities, optimized for efficient dataflow in post-training workflows.

<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696193102-a5654375-65a1-4e06-9c63-142b59df90b8.png" width="70%">
</p>


TransferQueue offers **fine-grained, sample-level** data management and **load-balancing** (on the way) capabilities, serving as a data gateway that decouples explicit data dependencies across computational tasks. This enables a divide-and-conquer approach, significantly simplifying the design of the algorithm controller.


<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696791245-fa7baf96-46af-4c19-8606-28ffadc4556c.png" width="70%">
</p>






<h2 id="updates">🔄 Updates</h2>

 - **Oct 21, 2025**: Official integration into verl is ready [verl/pulls/3649](https://github.com/volcengine/verl/pull/3649). Following PRs will optimize the single controller architecture by fully decoupling data & control flows.
 - **July 22, 2025**: We present a series of Chinese blogs on <a href="https://zhuanlan.zhihu.com/p/1930244241625449814">Zhihu 1</a>, <a href="https://zhuanlan.zhihu.com/p/1933259599953232589">2</a>.
 - **July 21, 2025**: We started an RFC on verl community [verl/discussions/2662](https://github.com/volcengine/verl/discussions/2662).
 - **July 2, 2025**: We publish the paper [AsyncFlow](https://arxiv.org/abs/2507.01663).



<h2 id="components">🧩 Components</h2>



### Control Plane: Panoramic Data Management  

In the control plane, `TransferQueueController` tracks the **production status** and **consumption status** of each training sample as metadata. When all the required data fields are ready (i.e., written to the `TransferQueueStorage`), we know that this data sample can be consumed by downstream tasks. 

For consumption status, we record the consumption records for each computational task (e.g., `generate_sequences`, `compute_log_prob`, etc.). Therefore, even when different computation tasks require the same data field, they can consume the data independently without interfering with each other.


<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696820173-456c1784-42ba-40c8-a292-2ff1401f49c5.png" width="70%">
</p>


> In the future, we plan to support **load-balancing** and **dynamic batching** capabilities in the control plane. Additionally, we will support data management for disaggregated frameworks where each rank manages the data retrieval by itself, rather than coordinated by a single controller.

### Data Plane: Distributed Data Storage

In the data plane, `TransferQueueStorageSimpleUnit` serves as a naive storage unit based on CPU memory, responsible for the actual storage and retrieval of data. Each storage unit can be deployed on a separate node, allowing for distributed data management.

`TransferQueueStorageSimpleUnit` employs a 2D data structure as follows:

- Each row corresponds to a training sample, assigned a unique index within the corresponding global batch.
- Each column represents the input/output data fields for computational tasks.

This data structure design is motivated by the computational characteristics of the post-training process, where each training sample is generated in a relayed manner across task pipelines. It provides an accurate addressing capability, which allows fine-grained, concurrent data read/write operations in a streaming manner.

<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696805154-3817011f-84e6-40d0-a80c-58b7e3e5f6a7.png" width="70%">
</p>


> In the future, we plan to implement a **general storage abstraction layer** to support various storage backends. Through this abstraction, we hope to integrate high-performance storage solutions such as [MoonCakeStore](https://github.com/kvcache-ai/Mooncake) to support device-to-device data transfer through RDMA, further enhancing data transfer efficiency for large-scale data.


### User Interface: Asynchronous & Synchronous Client


The interaction workflow of TransferQueue system is as follows:

1. A process sends a read request to the `TransferQueueController`.
2. `TransferQueueController` scans the production and consumption metadata for each sample (row), and dynamically assembles a micro-batch metadata according to the load-balancing policy. This mechanism enables sample-level data scheduling.
3. The process retrieves the actual data from distributed storage units using the metadata provided by the controller.

To simplify the usage of TransferQueue, we have encapsulated this process into `AsyncTransferQueueClient` and `TransferQueueClient`. These clients provide both asynchronous and synchronous interfaces for data transfer, allowing users to easily integrate TransferQueue into their framework.


> In the future, we will provide a `StreamingDataLoader` interface for disaggregated frameworks as discussed in [RFC#2662](https://github.com/volcengine/verl/discussions/2662). Leveraging this abstraction, each rank can automatically get its own data like `DataLoader` in PyTorch. The TransferQueue system will handle the underlying data scheduling and transfer logic caused by different parallelism strategies, significantly simplifying the design of disaggregated frameworks.


<h2 id="show-cases">🔥 Showcases</h2>

### General Usage

The primary interaction points are `AsyncTransferQueueClient` and `TransferQueueClient`, serving as the communication interface with the TransferQueue system.

Core interfaces:

- (async_)get_meta(data_fields: list[str], batch_size:int, global_step:int, get_n_samples:bool, task_name:str) -> BatchMeta
- (async_)get_data(metadata:BatchMeta) -> TensorDict
- (async_)put(data:TensorDict, metadata:BatchMeta, global_step)
- (async_)clear(global_step: int)


We will soon release a detailed tutorial and API documentation.


### Collocated Example

#### verl
The primary motivation for integrating TransferQueue to verl now is to **alleviate the data transfer bottleneck of the single controller `RayPPOTrainer`**. Currently,  all `DataProto` objects must be routed through `RayPPOTrainer`, resulting in a single point bottleneck of the whole post-training system. 

![verl_dataflow_DataProto](https://cdn.nlark.com/yuque/0/2025/jpeg/23208217/1758704289414-bcc54228-716b-4d4a-ad3b-f9ace6d10fcf.jpeg)

Leveraging TransferQueue, we separate experience data transfer from metadata dispatch by

- Replacing `DataProto` with `BatchMeta` (metadata) and `TensorDict` (actual data) structures
- Preserving verl's original Dispatch/Collect logic via BatchMeta (maintaining single-controller debuggability)
- Accelerating data transfer by TransferQueue's distributed storage units

![verl_dataflow_TransferQueue](https://cdn.nlark.com/yuque/0/2025/jpeg/23208217/1758704301666-0807dc06-766c-4a2d-9cde-889a6bb56b34.jpeg)


You may refer to the [recipe](https://github.com/TransferQueue/TransferQueue/tree/dev/recipe/simple_use_case), where we mimic the verl usage in both async & sync scenarios. Official integration to verl is on the way.


### Disaggregated Example

Work in progress :) 


<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696840817-14ba4c3b-b96e-4390-ac7c-4ecf7b8c0ac3.png" width="70%">
</p>


<h2 id="quick-start">🚀 Quick Start</h2>


### Use Python package
We will soon release the Python package on PyPI. 

### Build wheel package from source code

Follow these steps to build and install:
1. Retrieve source code from GitHub repo
   ```bash
   git clone https://github.com/TransferQueue/TransferQueue/
   cd TransferQueue
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Build and install
   ```bash
   python -m build --wheel
   pip install dist/*.whl
   ```


<h2 id="milestones"> 🛣️ RoadMap</h2>

- [ ] Support data rewrite for partial rollout & agentic post-training
- [x] Provide a general storage abstraction layer `TransferQueueStorageManager` to manage distributed storage units, which simplifies `Client` design and makes it possible to introduce different storage backends ([PR66](https://github.com/TransferQueue/TransferQueue/pull/66))
- [ ] Provide a `KVStorageManager` to cover all the KV based storage backends
- [ ] Support topic-based data partitioning to maintain train/val/test data simultaneously
- [ ] Release the first stable version through PyPI
- [ ] Support disaggregated framework (each rank retrieves its own data without going through a centralized node)
- [ ] Provide a `StreamingDataLoader` interface for disaggregated framework
- [ ] Support load-balancing and dynamic batching
- [ ] Support high-performance storage backends for RDMA transmission (e.g., [MoonCakeStore](https://github.com/kvcache-ai/Mooncake), [Ray Direct Transport](https://docs.ray.io/en/master/ray-core/direct-transport.html)...)
- [ ] High-performance serialization and deserialization
- [ ] More documentation, examples and tutorials

<h2 id="citation">📑 Citation</h2>
Please kindly cite our paper if you find this repo is useful:

```bibtex
@article{han2025asyncflow,
  title={AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training},
  author={Han, Zhenyu and You, Ansheng and Wang, Haibo and Luo, Kui and Yang, Guang and Shi, Wenqi and Chen, Menglong and Zhang, Sicheng and Lan, Zeshun and Deng, Chunshi and others},
  journal={arXiv preprint arXiv:2507.01663},
  year={2025}
}
```
