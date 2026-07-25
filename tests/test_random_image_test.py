from pathlib import Path

from scripts.random_image_test import find_jpegs, safe_name


def test_find_jpegs_is_case_insensitive_and_recursive(tmp_path: Path):
    (tmp_path / "one.JPG").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "two.jpeg").write_bytes(b"")
    (nested / "ignore.png").write_bytes(b"")

    assert find_jpegs(tmp_path, recursive=True) == [nested / "two.jpeg", tmp_path / "one.JPG"]
    assert find_jpegs(tmp_path, recursive=False) == [tmp_path / "one.JPG"]


def test_safe_name_makes_output_folder_names_portable():
    assert safe_name("portrait / test.jpg") == "portrait___test_jpg"
