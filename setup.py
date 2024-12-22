from setuptools import setup, find_packages

setup(
    name="score_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "nltk>=3.8.0",
        "rouge>=1.0.1",
        "radon>=5.1.0",
        "sympy>=1.12",
        "typing-extensions>=4.5.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Self-Correcting Language Model with Reinforcement Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Self-Correcting-LLM--Reinforcement-Learning-",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 