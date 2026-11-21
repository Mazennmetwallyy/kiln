import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run tests that need real model weights")
    parser.addoption("--runmps", action="store_true", help="run tests that need an Apple GPU (MPS)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real model weights (downloads)")
    config.addinivalue_line("markers", "mps: needs an Apple GPU")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for mark, opt in (("slow", "--runslow"), ("mps", "--runmps")):
            if mark in item.keywords and not config.getoption(opt):
                item.add_marker(pytest.mark.skip(reason=f"needs {opt}"))
