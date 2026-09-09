def pytest_addoption(parser):
    parser.addoption(
        "--run-hf-data",
        action="store_true",
        default=False,
        help="Run tests that query the Hugging Face dataset repos over the network.",
    )
