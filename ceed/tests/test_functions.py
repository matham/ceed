import pytest


@pytest.fixture
def func_factory():
    from ceed.function import FunctionFactoryBase
    return FunctionFactoryBase()


def function_factory(func_factory):
    assert not func_factory.funcs_cls
    assert not func_factory.funcs_user
    assert not func_factory.funcs_inst
    assert not func_factory.funcs_inst_default
