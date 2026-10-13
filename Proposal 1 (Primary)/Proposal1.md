# High-Performance Image Preprocessing for Deep Learning using AVX2 Assembly

## Abstract of your proposed project

The project implements a SIMD data augmentation pipeline designed to accelerate the preparation of training datasets for Convolutional Neural Networks (CNNs) in computer vision tasks. Based on the sequential data augmentation methodology proposed by Sengupta et al. (2023) which comprises rescaling, rotating, flipping and grayscale conversion, this project accelerates two preprocessing steps: vertical flipping and grayscale conversion. By replacing Python library calls with specialized x86-64 assembly kernels using AVX2 SIMD instructions, this project demonstrated how low-level hardware optimizations can speed up data preprocessing pipelines.

## Description of your proposed project
### What are the inputs to your project?
* A scalar variable representing the total number of pixels or elements to process
* Input arrays representing raw RGB images

### Propose process
1. What processes will be parallelized using SIMT (GPU/CUDA) or SIMD-based AVX/AVX2/AVX512 assembly project?    
     
    a. Vertical Flipping: Inverting the matrix along the vertical axis to create a geometric variation of the original image

    b. Grayscale Conversion: Reducing image color dimensionality from 3 (RGB) to 1 by computing for the average of channel intensities

2. What existing implementations have been done with the proposed process (i.e., related works)? How would it differ from your proposed implementation if it has been parallelized?

Common deep learning pipelines implement these steps using high-level frameworks like PyTorch, torchvision, OpenCV, and Keras. In PyTorch specifically, flipping can be represented as an O(1) operation (image[::-1, :]), which alters tensor index metadata rather than actually reordering memory, which is often required in deep learning. Moreover, invoking these Python-wrapped functions incurs runtime overhead from interpreter dispatch, dynamic type checks, and intermediate tensor allocations.

Our implementation differs by executing direct memory arrangements via AVX2 assembly kernels. We aim to maximize memory throughput and minimize processing latency on CPU platforms. 

3. Discuss plans on how to parallelize your proposed process.

* **Grayscale conversion**: Load packed single-precision RGB pixel channels into 128-bit XMM and 256-bit YMM registers. Compute the grayscale average across 4 (XMM) or 8 (YMM) pixels simultaneously in a single clock cycle using the formula: Imgray(x,y)=13(ImR(x,y)+ImG(x,y)+ImB(x,y)) 

* **Vertical Flipping**: Immediately write the computed grayscale vector directly to its mirrored row position in the destination buffer, completing both the mathematical conversion and the vertical flip in a single memory pass.

4. What other processes are in the project but will not be parallelized

* Rescaling and Rotating
* Dataset reading and file I/O operations
* CNN training and classification workflows

### What are the outputs of your project?

- Grayscale image arrays physically mirrored along the vertical axis
- Benchmarks records
    - Python/PyTorch and C implementations
    - Our AVX2 implementation


## Proposed target implementation platform of the proposed project
* x86-64 Architecture + AVX2
* Visual Studio 2022
* Python and C 

## Mid-project milestones of the proposed project
* Establish mathematical ground truth and performance baselines by developing an un-fused sequential C program alongside a standard PyTorch implementation.
* Develop a scalar x86-64 assembly kernel to validate the fused memory pass, and upgrade it to utilize 128-bit XMM and 256-bit YMM registers for full data-level parallelization.
* Implement boundary handling for non-divisible image dimensions, and benchmark the final AVX2 throughput against the C and PyTorch baselines.

## References: 
* PyTorch. (n.d.). Transforming and augmenting images. Torchvision 0.13 documentation. https://docs.pytorch.org/vision/0.13/transforms.html
* Sengupta, P., Mehta, A., & Rana, P. S. (2023). Enhancing performance of deep learning models with a novel data augmentation approach. IEEE. https://ieeexplore.ieee.org/document/10308298

