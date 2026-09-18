# Resource-Constrained Distributed Computing

**Enabling Data-Parallel Model Training on Consumer Laptops**

## Abstract
Modern machine learning (ML) workloads are highly compute-intensive, yet dedicated high-performance clusters and cloud-based GPU resources remain expensive and largely inaccessible to students or small research groups. 

This project investigates the feasibility of pooling a network of standard, unmodified consumer laptops into a functional distributed computing cluster capable of efficiently executing resource-intensive ML training. We have developed a distributed architecture utilizing message-passing to coordinate parallel execution across independent physical machines. 

Building upon this foundational framework, we implemented a data-parallel strategy for ML model training, where training datasets are partitioned across the network, gradients are computed locally on each machine, and model parameters are periodically synchronized with a primary coordinating node.

Ultimately, this work demonstrates that meaningful distributed computing and scalable AI model training can be achieved using constrained, heterogeneous commodity hardware without relying on specialized server infrastructure.

## Key Features
- **Distributed Architecture:** Coordinates parallel execution across independent physical machines.
- **Data Parallelism:** Partitions datasets across multiple workers to compute gradients locally on data shards.
- **Message Passing & Synchronization:** Utilizes a primary coordinating node to synchronize model parameters, mirroring the Bulk Synchronous Parallel (BSP) superstep structure.
- **Heterogeneity & Straggler Mitigation:** Designed to operate efficiently on non-dedicated, heterogeneous consumer hardware with varying CPU capabilities, memory bandwidth, and background loads.

## Repository Contents
- `Master.py` / `GridApp.py`: Primary coordinating scripts for the distributed network.
- `Payload.py` / `Generate.py`: Utilities for data partitioning, generation, and payload management.
- `Script.py`: Worker node execution logic.
- `Data.txt`: Sample dataset for distributed training.

## Authors
- **Suyash Srivastava** (25BCE5366) - B.Tech Computer Science Engineering, VIT Chennai
- **Shanil Singh** (25BCE5380) - B.Tech Computer Science Engineering, VIT Chennai

## References
This project is built upon foundational distributed computing paradigms, specifically inspired by:
1. Chen, C., et al. (2023). "Accelerating distributed learning in non-dedicated environments." *IEEE Transactions on Cloud Computing*.
2. Hu, H., et al. (2020). "Distributed machine learning through heterogeneous edge systems." *AAAI*.
3. Narayanan, D., et al. (2019). "PipeDream: Generalized pipeline parallelism for DNN training." *SOSP*.
4. Thakur, R., et al. (2005). "Optimization of collective communication operations in MPICH." *IJHPCA*.
5. Valiant, L. G. (1990). "A bridging model for parallel computation." *Communications of the ACM*.
