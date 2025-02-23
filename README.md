# llama3.1-1B-installation
- Download Llama 3.2 1B model from hugging face  [![Llama-3.2-1B](https://img.shields.io/badge/Llama_3.2_1B-hugging_face-yellow)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
**- One of the Prerequisites You need to have around 10GB of space on your disk.**
- Extend C drive Storage space  [![Link](https://img.shields.io/badge/Youtube-red)](https://www.youtube.com/watch?v=0gLgCupVvVI)

## Setup-NVIDIA-GPU
- Step 1: NVIDIA Video Driver  [![Link](https://img.shields.io/badge/nVIDIA_GPU_Driver-76B900)](https://www.nvidia.com/en-us/drivers/)
- Step 2: Visual Studio C++  [![Visual_Studio Community](https://img.shields.io/badge/Visual_Studio_Community-purple)](https://visualstudio.microsoft.com/vs/community/)
- Step 3: Anaconda/Miniconda  [![Link](https://img.shields.io/badge/Anaconda-3EB049)](https://www.anaconda.com/download/success)
- Step 4: CUDA Toolkit [![Link](https://img.shields.io/badge/CUDA_Toolkit_Archive-76B900)](https://developer.nvidia.com/cuda-toolkit-archive)
- Step 5: cuDNN [![Link](https://img.shields.io/badge/cuDNN_Archive-76B900)](https://developer.nvidia.com/rdp/cudnn-archive)
- Step 6: Install PyTorch [![Link](https://img.shields.io/badge/pytorch-orange)](https://pytorch.org/get-started/locally/)
- Finally run the following script to test your GPU
    ```python
    import torch

    print("Number of GPU: ", torch.cuda.device_count())
    print("GPU Name: ", torch.cuda.get_device_name())


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    ```

  
 
