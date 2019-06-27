"""Load beem configuration data."""

import os.path
import pytoml

from dcss import bot_services

class BotConfig():
    """Base class for TOML config parsing for bots."""

    def __init__(self, path):
        self.data = {}
        self.path = path

    def __getattr__(self, name):
        return self.data[name]

    def load(self):
        """Read the main TOML configuration data from self.path and check that
        the configuration is valid."""

        if not os.path.exists(self.path):
            self.error("Couldn't find file!")

        try:
            config_fh = open(self.path, "r")
        except EnvironmentError as e:
            self.error("Couldn't open file: ({})".format(e.strerror))
        else:
            try:
                self.data = pytoml.load(config_fh)
            except pytoml.TomlError as e:
                self.error("Couldn't parse TOML file {} at line {}, col {}: "
                           "{}".format(e.filename, e.line, e.col, e.message))
            finally:
                config_fh.close()
