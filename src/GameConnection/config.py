import os.path
import pytoml

ConfigFile = "src/GameConnection/config.toml"

class BotConfig():
    def __init__(self):
        self.data = {}

    def __getattr__(self, name):
        return self.data[name]

    def load(self):
        config_fh = open(ConfigFile, "r")
        self.data = pytoml.load(config_fh)
        config_fh.close()
