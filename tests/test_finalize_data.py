import hashlib
from pathlib import Path
import tempfile
import unittest

from finalize_data import sha256


class FinalizeDataTests(unittest.TestCase):
    def test_text_fingerprints_are_independent_of_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = hashlib.sha256(b'{"config": true}\n').hexdigest()
            lf_file = root / "lf.json"
            crlf_file = root / "crlf.json"
            lf_file.write_bytes(b'{"config": true}\n')
            crlf_file.write_bytes(b'{"config": true}\r\n')

            self.assertEqual(sha256(lf_file), expected)
            self.assertEqual(sha256(crlf_file), expected)

    def test_binary_fingerprints_remain_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")

            self.assertEqual(sha256(image), hashlib.sha256(image.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
