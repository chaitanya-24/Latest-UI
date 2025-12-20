from setuptools import setup, find_packages

setup(
    name="iagentops",
    version="0.1.0",
    description="iAgentOps SDK for instrumentation",
    author="Chaitanya",
    packages=find_packages(include=["iagentops", "iagentops.*"]),
    install_requires=[
        "opentelemetry-api",
        "opentelemetry-sdk",
        "wrapt",
        "asyncio",
        "langchain",
        "langgraph",
        "pip-system-certs",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)




