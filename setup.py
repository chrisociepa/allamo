import os
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension, 
    CUDAExtension,
)

this_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    CUDAExtension(
        name='grkan_cuda_lib',
        sources=[
            "csrc/grkan/interface.cpp",
            "csrc/grkan/grkan_kernel.cu"
        ],
        include_dirs=[os.path.join(this_dir, "csrc/grkan")],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],
        },
    )
]

setup(name='allamo',
      version='5.0.0',
      author='Krzysztof (Chris) Ociepa',
      author_email='chris@azurro.pl',
      packages=['allamo'],
      description='Simple, hackable and fast implementation for training/finetuning medium-sized LLaMA-based models',
      license='MIT',
      python_requires=">=3.8",
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension},
      install_requires=[
            'torch',
            'numpy',
            'joblib',
            'wandb'
      ],
)
