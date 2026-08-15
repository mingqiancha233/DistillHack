from setuptools import setup, find_packages

setup(
    name="distill_hack_nas",
    version="0.1.0",
    description="A NAS framework using implicit distillation via biased random numbers.",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "peft>=0.8.2",
        "bitsandbytes>=0.41.0",
        "optuna>=3.5.0",
        "openai>=1.12.0",
        "anthropic>=0.19.0",
        "pydantic>=2.6.0",
        "tqdm"
    ],
)
