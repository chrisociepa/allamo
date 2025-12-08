from importlib.metadata import version

try:
    __version__ = version("allamo")
except Exception as e:
    __version__ = "0.0.0+unknown"
