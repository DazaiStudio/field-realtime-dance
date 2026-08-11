import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osc_viewer


class TestDefaultOscTargets(unittest.TestCase):
    """Targets live in memory and reset on every restart, so the defaults are
    the only thing standing between a restart and OSC silently going nowhere."""

    def test_defaults_to_both_show_machines(self):
        with mock.patch.dict(os.environ, {"FIELD_OSC_TARGETS": ""}, clear=False):
            targets = osc_viewer.default_osc_targets()
        self.assertEqual([t["host"] for t in targets], ["10.0.0.102", "10.0.0.103"])
        self.assertTrue(all(t["port"] == 9000 for t in targets))
        self.assertTrue(all(t["enabled"] for t in targets))

    def test_first_target_keeps_the_default_id(self):
        with mock.patch.dict(os.environ, {"FIELD_OSC_TARGETS": ""}, clear=False):
            targets = osc_viewer.default_osc_targets()
        self.assertEqual(targets[0]["id"], "default")
        self.assertNotEqual(targets[1]["id"], targets[0]["id"])

    def test_env_override_wins(self):
        with mock.patch.dict(os.environ,
                             {"FIELD_OSC_TARGETS": "192.168.1.5:9100, 192.168.1.6:9200"},
                             clear=False):
            targets = osc_viewer.default_osc_targets()
        self.assertEqual([(t["host"], t["port"]) for t in targets],
                         [("192.168.1.5", 9100), ("192.168.1.6", 9200)])

    def test_empty_osc_targets_form_field_yields_the_defaults(self):
        with mock.patch.dict(os.environ, {"FIELD_OSC_TARGETS": ""}, clear=False):
            targets = osc_viewer.parse_osc_targets("", "10.0.0.102", 9000)
        self.assertEqual(len(targets), 2)

    def test_explicit_targets_are_still_honoured(self):
        targets = osc_viewer.parse_osc_targets(
            '[{"host": "127.0.0.1", "port": 9000}]', "10.0.0.102", 9000)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["host"], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
