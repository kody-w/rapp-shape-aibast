"""Contract tests: the local storage shim vs CommunityRAPP's real API.

The shim's one job (docstring line 2, repo CLAUDE.md) is "cloud agents work
locally". Review found the entire file API was arity-incompatible with the
cloud's (directory_name, file_name[, content]) signatures — every RAR registry
agent using files TypeError'd through the standard install flow.

communityrapp_storage_api.json is an AST-extracted snapshot of the cloud
class (RAPP/rapp_swarm/utils/azure_file_storage.py). If the cloud API moves,
re-extract the snapshot and adapt the shim in the same change.
"""
import inspect
import json
import os
import shutil
import tempfile
import unittest

import local_storage
from local_storage import AzureFileStorageManager

_SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "communityrapp_storage_api.json")


class _ShimTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="shimtest-")
        self._orig = local_storage._DATA_DIR
        local_storage._DATA_DIR = self._tmp
        self.mgr = AzureFileStorageManager()

    def tearDown(self):
        local_storage._DATA_DIR = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestCloudSignatureParity(unittest.TestCase):
    """Every public cloud method must exist on the shim and bind every call
    shape a cloud-authored agent can produce (positional AND keyword)."""

    @classmethod
    def setUpClass(cls):
        with open(_SNAPSHOT, encoding="utf-8") as f:
            cls.api = json.load(f)["methods"]

    def test_every_cloud_method_binds(self):
        for method, spec in self.api.items():
            with self.subTest(method=method):
                self.assertTrue(hasattr(AzureFileStorageManager, method),
                                f"shim is missing cloud method {method}()")
                sig = inspect.signature(getattr(AzureFileStorageManager, method))
                dummies = {name: object() for name in spec["required"]}
                # Cloud agents call positionally...
                sig.bind(None, *dummies.values())
                # ...and by keyword, using the cloud's parameter names.
                sig.bind(None, **dummies)


class TestCloudCallShapesWork(_ShimTestBase):
    """Exercise the cloud call shapes end-to-end against local disk."""

    def test_two_arg_file_roundtrip(self):
        self.assertTrue(self.mgr.write_file("reports", "summary.txt", "hello cloud"))
        self.assertTrue(self.mgr.file_exists("reports", "summary.txt"))
        self.assertEqual(self.mgr.read_file("reports", "summary.txt"), "hello cloud")
        props = self.mgr.get_file_properties("reports", "summary.txt")
        self.assertEqual(props["name"], "summary.txt")
        self.assertEqual(props["size"], len(b"hello cloud"))
        self.assertTrue(self.mgr.delete_file("reports", "summary.txt"))
        self.assertFalse(self.mgr.file_exists("reports", "summary.txt"))

    def test_binary_content_and_binary_read(self):
        payload = bytes(range(256))
        self.assertTrue(self.mgr.write_file("blobs", "data.zip", payload))
        # .zip is a known-binary extension → read_file returns bytes (cloud rule)
        self.assertEqual(self.mgr.read_file("blobs", "data.zip"), payload)
        self.assertEqual(self.mgr.read_file_binary("blobs", "data.zip"), payload)

    def test_list_files_entries_satisfy_both_client_styles(self):
        self.mgr.write_file("docs", "a.txt", "x")
        entries = self.mgr.list_files("docs")
        self.assertEqual([e.name for e in entries], ["a.txt"])  # cloud style
        self.assertIn("a.txt", entries)                          # legacy style

    def test_list_files_auto_creates_missing_directory(self):
        self.assertEqual(self.mgr.list_files("brand-new-dir"), [])
        self.assertEqual(self.mgr.list_files("brand-new-dir", auto_create=False), [])

    def test_ensure_directory_exists_nested(self):
        self.assertTrue(self.mgr.ensure_directory_exists("a/b/c"))
        self.assertTrue(self.mgr.write_file("a/b/c", "deep.txt", "d"))
        self.assertFalse(self.mgr.ensure_directory_exists(""))

    def test_missing_file_contract(self):
        self.assertIsNone(self.mgr.read_file("nowhere", "gone.txt"))
        self.assertIsNone(self.mgr.read_file_binary("nowhere", "gone.txt"))
        self.assertIsNone(self.mgr.get_file_properties("nowhere", "gone.txt"))
        self.assertFalse(self.mgr.delete_file("nowhere", "gone.txt"))

    def test_traversal_never_raises_only_fails(self):
        # Cloud contract: file ops log and return failure values, never raise.
        self.assertFalse(self.mgr.write_file("..", "escape.txt", "x"))
        self.assertIsNone(self.mgr.read_file("..", "escape.txt"))
        self.assertFalse(self.mgr.file_exists("..", "escape.txt"))
        self.assertEqual(self.mgr.list_files(".."), [])


class TestSetMemoryContextContract(_ShimTestBase):
    VALID_GUID = "0f1e2d3c-4b5a-6978-8695-a4b3c2d1e0f9"

    def test_falsy_and_marker_use_shared_and_succeed(self):
        for value in (None, "", AzureFileStorageManager.DEFAULT_MARKER_GUID):
            with self.subTest(value=value):
                self.assertTrue(self.mgr.set_memory_context(value))
                self.assertIsNone(self.mgr.current_guid)

    def test_valid_guid_isolates(self):
        self.mgr.set_memory_context(None)
        self.mgr.write_json({"shared": True})
        self.assertTrue(self.mgr.set_memory_context(self.VALID_GUID))
        self.mgr.write_json({"user": True})
        self.assertEqual(self.mgr.read_json(), {"user": True})
        self.mgr.set_memory_context(None)
        self.assertEqual(self.mgr.read_json(), {"shared": True})

    def test_invalid_guid_falls_back_to_shared_returns_false_never_raises(self):
        self.mgr.set_memory_context(None)
        self.mgr.write_json({"shared": True})
        for bad in ("user-abc", "a/../b", "guid-123", "CON", "user ", 123, False):
            with self.subTest(bad=bad):
                result = self.mgr.set_memory_context(bad)
                if bad is False:  # falsy → shared, True (cloud: `if not guid`)
                    self.assertTrue(result)
                else:
                    self.assertFalse(result)
                self.assertIsNone(self.mgr.current_guid)
                self.assertEqual(self.mgr.read_json(), {"shared": True})

    def test_cloud_and_legacy_keyword_spellings(self):
        self.assertTrue(self.mgr.set_memory_context(guid=self.VALID_GUID))
        self.assertEqual(self.mgr.current_guid, self.VALID_GUID)
        self.assertTrue(self.mgr.set_memory_context(user_guid=self.VALID_GUID))
        self.assertEqual(self.mgr.current_guid, self.VALID_GUID)


class TestLegacyLocalShapesStillWork(_ShimTestBase):
    """User-authored agents from older installs used single-path signatures —
    upgrades preserve those agents, so the shapes must keep working."""

    def test_legacy_single_path_roundtrip(self):
        self.assertTrue(self.mgr.write_file("test/hello.txt", "world"))
        self.assertTrue(self.mgr.file_exists("test/hello.txt"))
        self.assertEqual(self.mgr.read_file("test/hello.txt"), "world")
        self.assertIn("hello.txt", self.mgr.list_files("test"))
        self.assertTrue(self.mgr.delete_file("test/hello.txt"))
        self.assertFalse(self.mgr.file_exists("test/hello.txt"))


if __name__ == "__main__":
    unittest.main()
