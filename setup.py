from setuptools import setup, find_packages

setup(
    name="sac-diffusion-driving",
    version="0.1.0",
    author="Wujiahao",
    description="SAC-based End-to-End Diffusion Driving Policy for Autonomous Navigation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sac-diffusion-driving",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.9",
    
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "gymnasium>=0.28.1",
        "stable-baselines3>=2.0.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.0",
        "wandb>=0.15.0",
        "einops>=0.6.1",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "vis": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "pyvista>=0.41.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "train-bc=scripts.train_bc:main",
            "train-sac=scripts.train_sac_diffusion:main",
            "evaluate=scripts.evaluate:main",
            "deploy=scripts.deploy_to_robot:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
