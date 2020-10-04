import importlib


def get_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls
