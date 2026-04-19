from setuptools import setup, find_packages

setup(
    name="cipher",
    version="0.1.0",
    description="ciPHer: Capsule Interaction Prediction of Phage-Host Relationships",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "torch": ["torch", "tqdm"],
        "xgboost": ["xgboost", "scikit-learn"],
        "viz": ["matplotlib"],
        "test": ["pytest"],
        "all": ["torch", "tqdm", "xgboost", "scikit-learn", "matplotlib", "pytest"],
    },
    entry_points={
        "console_scripts": [
            "cipher-evaluate=cipher.evaluation.runner:main",
            "cipher-train=cipher.cli.train_runner:main",
            "cipher-analyze=cipher.cli.analyze_runner:main",
        ],
    },
)
