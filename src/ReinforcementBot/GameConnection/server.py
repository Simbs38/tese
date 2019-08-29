"""beem: A multi-user chat bot that can relay queries to the IRC
knowledge bots for DCSS from WebTiles chat."""

import argparse

import asyncio
if hasattr(asyncio, "async"):
    ensure_future = asyncio.async
else:
    ensure_future = asyncio.ensure_future

import functools
import logging
import os
import signal
import sys
import traceback
import webtiles

from config import BotConfig
from webtiles import WebTilesManager

# Will be configured by beem_server after the config is loaded.
_log = logging.getLogger()

class BeemServer:
    def __init__(self):
        self.webtiles_task = None
        self.loop = asyncio.get_event_loop()
        self.conf = BotConfig()
        self.conf.load()
        self.load_webtiles()

    def load_webtiles(self):
        wtconf = self.conf.webtiles
        self.webtiles_manager = WebTilesManager(wtconf)

        if wtconf.get("watch_username"):
            user_data = bot_db.get_user_data(wtconf["watch_username"])
            if not user_data:
                user_data = bot_db.register_user(wtconf["watch_username"])
            if not user_data["subscription"]:
                bot_db.set_user_field(wtconf["watch_username"], "subscription",
                        1)

    def start(self):
        def do_exit(signame):
            is_error = True if signame == "SIGTERM" else False
            msg = "Shutting down server due to signal: {}".format(signame)
            if is_error:
                _log.error(msg)
            else:
                _log.info(msg)
            self.stop(is_error)

        for signame in ("SIGINT", "SIGTERM"):
            self.loop.add_signal_handler(getattr(signal, signame),
                                           functools.partial(do_exit, signame))
        self.loop.run_until_complete(self.process())
        self.loop.close()

    def stop(self, is_error=False):
        print("Stopping beem server.")
        if self.webtiles_task and not self.webtiles_task.done():
            self.webtiles_task.cancel()

    @asyncio.coroutine
    def process(self):
        tasks = []

        self.webtiles_task = ensure_future(self.webtiles_manager.start())
        tasks.append(self.webtiles_task)
        yield from asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        yield from self.webtiles_manager.disconnect()