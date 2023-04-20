from setuptools import setup, find_packages


setup(
    name='popresample',
    version='0.1',
    url='https://github.com/ChristianAdamcewicz/popresample',
    author = 'Christian Adamcewicz',
    author_email = 'christian.adamcewicz@monash.edu',
    description = 'Importance sampler for resampling gwpopulation results',
    packages=find_packages(exclude=["example"]),
    package_dir={"popresample": "popresample"},
    entry_points={
        "console_scripts": ["popresample=popresample.main:run"]},
    install_requires=["bilby", "bilby-pipe", "gwpopulation", "numpy", "scipy", "tqdm"],
    python_requires=">=3.9",
)
