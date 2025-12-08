from setuptools import setup

setup(name='allamo',
      version='7.0.0',
      author='Krzysztof (Chris) Ociepa',
      packages=['allamo'],
      description='Simple, hackable and fast implementation for training and finetuning language models',
      license='MIT',
      install_requires=[
            'torch',
            'numpy',
            'joblib',
            'wandb'
      ],
)
