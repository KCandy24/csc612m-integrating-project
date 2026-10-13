# Fused CUDA Kernel for High-Throughput Vector Search 

## Abstract of your proposed project
This project proposes a single-pass, fused custom CUDA kernel that evaluates metadata conditions and vector similarity concurrently within the same GPU execution thread. By performing a bitwise metadata check before floating-point arithmetic, the proposed kernel triggers warp-synchronized early exits, bypassing unnecessary memory fetches and compute cycles for non-matching documents. 

## Description of your proposed project
### What are the inputs to your project?
	
* High-dimensional floating-point array representing N document vectors stored in GPU VRAM
* Metadata bitmask array where individual bits represent categorical attributes
* Query vector along with a target search bitmask representing the required filtering constraints
	
### Propose process
1. What processes will be parallelized using SIMT (GPU/CUDA) or SIMD-based AVX/AVX2/AVX512 assembly project?

Warp-Synchronized Metadata Evaluation
Parallel bitwise and evaluate document metadata bitmasks across GPU threads

Vector Distance Arithmetic 
Fused dot-product and norm calculations for cosine similarity across vector dimensions

In-register score Reduction
Parallel reduction of partial vector dot-products using CUDA warp-shuffle primitives (__shfl_down_sync)

Parallel Top-K Candidates Selection
Parallel bitonic sort across thread blocks to isolage the top-K highest similarity scores 
		
2. What existing implementations have been done with the proposed process (i.e., related works)? How would it differ from your proposed implementation if it has been parallelized?

Some famous industrial vector databases and libraries like Meta AI FAISS and NVIDIA cUVS traditionally perform post-filtering (computing distances for all candidates first, then filtering on CPU/GPU in a separate kernel pass) or pre-filtering (filtering indices first, creating sparse memory lookups that break GPU memory coalescing)

Our implementation differs by fusing the relational metadata check directly into the CUDA vector execution thread. By checking bitmasks before performing vector dot-products, our kernel triggers warp-synchronized early exits. 

3. Discuss plans on how to parallelize your proposed process.
Data Layout Optimization: Transform input embeddings into a contiguous Structure of Arrays (SoA) format aligned to 128-byte boundaries to ensure coalesced memory reads across CUDA threads

Grid and Block Mapping: Launch a CUDA grid where 1 CUDA Thread block processes a chunk of document vectors, mapping individual threads to vector dimensions

Early-Exit bitmask evaluation: Thread 0 of each warp evaluates the required bitmask using bitwise AND. If all document candidates within a warp fail the filter, they are immediately “returned”, skipping the floating-point execution entirely

Fused Dot-Product & Warp Reduction: For passing vectors, threads compute element-wise multiplication in parallel registers and sum them across the warp using CUDA intrinsics without using slow global memory

Top-K Bitonic Sort: Leverages bitonic sorting to isolate highest scores before returning results to the host CPU

4. What other processes are in the project but will not be parallelized?

* CPU memory allocation, file I/O operations
* Initial encoding/packing of categorical metadata strings into 32-bit integer bitmasks
* Formatting/Printing of retrieved document IDs

### What are the outputs of your project?
- An array containing the top-K document IDs and their corresponding similarity scores using the custom CUDA kernel
- Performance benchmarking metrics
    - Query Latency
    - Effective Memory Bandwidth
    - Speedup Factors

## Proposed target implementation platform of the proposed project
* NVIDIA GPU
* C/C++, CUDA

## Mid-project milestones of the proposed project
* Build document metadata parser/tokenizer for embedding: Use existing conversions to transform documents into dense arrays of embeddings. 
* Develop Fused Kernel design: Multi-threaded implementation of metadata and embedding filtering to the document data. Each thread checks documents from a shared document queue.
* Post-optimization: Handle "Warp Divergence" by clustering and optimize Top-K matches by using per thread arrays.

## References: 
* Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547. (Foundational paper for Meta FAISS GPU vector search principles). arXiv:1702.08734
* NVIDIA Corporation. (2024). NVIDIA cuVS: Vector Search and Clustering on GPUs. Developer Documentation & Tech Blog.
