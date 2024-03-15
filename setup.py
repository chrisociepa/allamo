from setuptools import setup

setup(name='allamo',
      version='2.2.2',
      author='Krzysztof (Chris) Ociepa',
      packages=['allamo'],
      description='Simple, hackable and fast implementation for training/finetuning medium-sized LLaMA-based models',
      license='MIT',
      install_requires=[
            'torch',
            'numpy',
      ],
)
