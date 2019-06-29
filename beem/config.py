"""Load beem configuration data."""

import os.path
import pytoml

class BotConfig():
    """Base class for TOML config parsing for bots."""

    def __init__(self, path):
        self.data = {}
        self.path = path

    def __getattr__(self, name):
        return self.data[name]

    def load(self):
        config_fh = open(self.path, "r")
        self.data = pytoml.load(config_fh)
        config_fh.close()
