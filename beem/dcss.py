"""IRC connection management for DCSS knowledge bots"""

import asyncio
if hasattr(asyncio, "async"):
    ensure_future = asyncio.async
else:
    ensure_future = asyncio.ensure_future

import base64
import irc.client
import logging
import jaraco.functools
import os
import signal
import re
import ssl
import string
import sys
import time
import traceback

_log = logging.getLogger()

# How long to wait in second for a query before ignoring any result from a bot
# and reusing the query ID. Sequell times out after 90s, but we use 180s since
# there may be instances where its response is slower.
_MAX_REQUEST_TIME = 180
# How long to wait after a connection failure before reattempting the
# connection.
_RECONNECT_TIMEOUT = 5

# Strings for services provided by DCSS bots. Used to match fields in the
# config andto indicate what type of query was performed.
bot_services = ["sequell", "monster", "git"]

# For extracting the single-character prefix from a Sequell !RELAY result.
_QUERY_PREFIX_CHARS = string.ascii_letters + string.digits
_query_prefix_regex = re.compile(r"^([a-zA-Z0-9])")

# Patterns for adding player and chat variables to Sequell queries.
_player_var_regex = re.compile(r"\$p(?=\W|$)|\$\{p\}")
_chat_var_regex = re.compile(r"\$chat(?=\W|$)|\$\{chat\}")

class IRCBot():
    """Coodrinate queries for a bot."""

    def __init__(self, manager, conf):
        self.manager = manager
        self.conf = conf

        self.init_services()

    def init_services(self):
        """Find any services we have in the config and create their regex
        pattern objects."""

        self.services = []
        self.service_patterns = {}

        for s in bot_services:
            field = "{}_patterns".format(s)
            if not field in self.conf:
                continue

            self.services.append(s)

            patterns = []
            for p in self.conf[field]:
                patterns.append(re.compile(p))
            self.service_patterns[s] = patterns

    def init_query_data(self):
        """Reset message and query tracking variables."""

        # A dict of query dicts keyed by the query id. Each entry holds info
        # about a DCSS query made to this bot so we can properly route the
        # result message when it's received back from the bot.
        self.queries = {}

        # A queue of query IDs used for synchronous services like monster and
        # git.
        self.queue = []

        # The query dict for the last query whose result we handled. Used to
        # route multiple part messages, primarily for monster queries, but
        # Sequell can sometimes send multiple messages in response to a single
        # query despite use of '!RELAY -n 1'
        self.last_answered_query = None

    def get_message_service(self, message):
        """Which type of service is this message intentded for? Returns None if
        none is found."""

        for s in self.services:
            for p in self.service_patterns[s]:
                if p.search(message):
                    return s

    def prepare_sequell_message(self, source, requester, query_id, message):
        """Format a message containing a query to send through Sequell's !RELAY
        command."""

        # Replace '$p' instances with the player's nick.
        if source.user:
            message = _player_var_regex.sub(source.get_dcss_nick(source.user),
                    message)

        # Replace '$chat' instances with a |-separated list of chat nicks.
        chat_nicks = source.get_chat_dcss_nicks(requester)
        if chat_nicks:
            message = _chat_var_regex.sub('@' + '|@'.join(chat_nicks), message)

        requester_nick = source.get_dcss_nick(requester)
        line_limit = ""
        if source.should_limit_sequell_lines(requester):
            line_limit = "-n 1 "

        return "!RELAY -nick {} -prefix {} {}{}".format(requester_nick,
                _QUERY_PREFIX_CHARS[query_id], line_limit, message)

    def expire_query_entries(self, current_time):
        """Expire query entries in the queries dict, the last returned query,
        and in the query id queue if they're too old relative to the given
        time."""

        last_query_age = None
        if self.last_answered_query:
            last_query_age = current_time - self.last_answered_query["time"]
        if last_query_age and last_query_age >= _MAX_REQUEST_TIME:
            self.last_answered_query = None

        for i in range(0, len(_QUERY_PREFIX_CHARS)):
            if i not in self.queries:
                if i in self.queue:
                    self.queue.remove(i)

            elif current_time - self.queries[i]["time"] >= _MAX_REQUEST_TIME:
                del self.queries[i]
                if i in self.queue:
                    self.queue.remove(i)

    def make_query_entry(self, source, username, message):
        """Find a query id available for use, recording the details of the
        requesting source and username as well as the time of the request in a
        dict that is stored in our dict of pending queries."""

        current_time = time.time()
        self.expire_query_entries(current_time)

        # Find an available query id.
        query_id = None
        for i in range(0, len(_QUERY_PREFIX_CHARS)):
            if (not i in self.queries
                    and (not self.last_answered_query
                        or i != self.last_answered_query['id'])):
                query_id = i
                break

        if query_id is None:
            raise Exception("too many queries in queue")

        query = {'id'           : query_id,
                 'requester'    : username,
                 'source_ident' : source.get_source_ident(),
                 'time'         : current_time,
                 'type'         : self.get_message_service(message)}
        self.queries[query_id] = query

        return query

    @asyncio.coroutine
    def send_query_message(self, source, requester, message):
        """Send a message containing a DCSS query to the bot."""

        query_entry = self.make_query_entry(source, requester, message)

        if 'sequell' in self.services:
            message = self.prepare_sequell_message(source, requester,
                    query_entry['id'], message)
        else:
            self.queue.append(query_entry['id'])

        yield from self.manager.send(self.conf["nick"], message)

    def get_message_query_id(self, message):
        """Get the originating query ID associated with the given IRC message
        recieved from the bot."""

        # This bot has Sequell, which means we assume it uses !RELAY and that
        # Sequell queries are the only type of queries the bot handles.
        if 'sequell' in self.services:
            match = _query_prefix_regex.match(message)
            if not match:
                _log.warning("DCSS: Received %s message with invalid "
                             "relay prefix: %s", self.conf["nick"], message)
                return

            return int(_QUERY_PREFIX_CHARS.index(match.group(1)))

        # The remain query types are non-Sequell and have no equivalent to
        # Sequell's !RELAY. These bots are synchronous, so the first entry in
        # the query id queue will be the relevant query id.
        else:
            if not len(self.queue):
                return

            return self.queue.pop(0)

    def get_message_query(self, message):
        """Find the query details we have based on the message or the queue."""

        self.expire_query_entries(time.time())

        query_id = self.get_message_query_id(message)
        # If we have no query information at all, return the query we last
        # answered, which may be None.
        if query_id is None:
            return self.last_answered_query

        # We have a query ID but no query data, meaning there's no unanswered
        # query corresponding to this ID. This can happen for Sequell when it
        # decides to send another line of response for a query even through we
        # use '-n 1' with '!RELAY'. In this case we can use the last answered
        # query if the IDs match.
        if not query_id in self.queries:
            if (self.last_answered_query
                and query_id == self.last_answered_query['id']):
                return self.last_answered_query

            else:
                _log.warning("DCSS: Unable to find query for %s result: %s",
                        self.conf['nick'], message)
                return

        self.last_answered_query = self.queries[query_id]
        del self.queries[query_id]
        return self.last_answered_query

