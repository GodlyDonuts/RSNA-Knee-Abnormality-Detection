"""Archive replay must preserve originals and reject mismatched ordering/bytes."""
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('colab_storage_continuation', HERE/'resume_bridge.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def info(name, size):
    entry = tarfile.TarInfo(name)
    entry.size, entry.mode = size, 0o600
    return entry


class StorageContinuation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'original.tar'
        with M.Archive(self.path) as out:
            out.addfile(info('a', 3), io.BytesIO(b'abc'))
            out.addfile(info('b', 2), io.BytesIO(b'de'))
        self.before = self.path.read_bytes()
        self.stat = self.path.stat()

    def tearDown(self):
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertEqual(self.path.stat().st_mtime_ns, self.stat.st_mtime_ns)
        self.temp.cleanup()

    def test_exact_replay_does_not_rewrite_or_reread_sources(self):
        with M.Archive(self.path) as out:
            out.addfile(info('a', 3), None)
            out.addfile(info('b', 2), None)

    def test_reordered_members_fail(self):
        with self.assertRaises(AssertionError), M.Archive(self.path) as out:
            out.addfile(info('b', 2), None)

    def test_changed_size_fails(self):
        with self.assertRaises(AssertionError), M.Archive(self.path) as out:
            out.addfile(info('a', 4), None)

    def test_missing_member_fails(self):
        with self.assertRaises(AssertionError), M.Archive(self.path) as out:
            out.addfile(info('a', 3), None)

    def test_extra_expected_member_fails(self):
        with self.assertRaises(StopIteration), M.Archive(self.path) as out:
            out.addfile(info('a', 3), None)
            out.addfile(info('b', 2), None)
            out.addfile(info('c', 1), None)

    def test_new_archive_written_once(self):
        fresh = self.path.with_name('new.tar')
        with M.Archive(fresh) as out:
            out.addfile(info('x', 4), io.BytesIO(b'data'))
        with tarfile.open(fresh) as original:
            self.assertEqual(original.extractfile('x').read(), b'data')

    def test_symlink_replay_is_rejected(self):
        link = self.path.with_name('symlink.tar')
        with tarfile.open(link, 'x') as out:
            member = info('x', 0)
            member.type = tarfile.SYMTYPE
            member.linkname = '/etc/passwd'
            out.addfile(member)
        with self.assertRaises(AssertionError), M.Archive(link) as out:
            out.addfile(info('x', 0), None)


if __name__ == '__main__':
    unittest.main()
