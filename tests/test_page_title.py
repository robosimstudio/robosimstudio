# SPDX-License-Identifier: Apache-2.0
"""Every page this package serves names its browser tab RoboSimStudio, not Viser."""

import unittest
from types import SimpleNamespace

from robosimstudio.sim.draw import PAGE_TITLE, title_page

TIER = "fast"


class TestTheTabIsNamed(unittest.TestCase):
    def test_the_title_is_queued_as_a_script_every_browser_runs(self) -> None:
        from viser import _messages

        queued = []
        server = SimpleNamespace(_websock_server=SimpleNamespace(queue_message=queued.append))
        title_page(server)
        self.assertEqual(PAGE_TITLE, "RoboSimStudio")
        (message,) = queued
        self.assertIsInstance(message, _messages.RunJavascriptMessage)
        self.assertEqual(message.source, 'document.title = "RoboSimStudio";')


if __name__ == "__main__":
    unittest.main()
