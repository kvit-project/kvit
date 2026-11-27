import subprocess

from unprefix.proxy import RepoIndex, layout, split_prompt


def test_layout_aligns_spans_and_following_text():
    pieces = [([1, 2, 3], False), ([7] * 5, True), ([4, 5], False), ([8] * 4, True)]
    ids, starts, crosses = layout(pieces, block_size=4, pad_id=0)
    assert ids == [1, 2, 3, 0, 7, 7, 7, 7, 7, 0, 0, 0, 4, 5, 0, 0, 8, 8, 8, 8]
    assert starts == [4, 16]
    assert crosses == [12]


def test_layout_skips_spans_shorter_than_a_block():
    ids, starts, crosses = layout([([1], False), ([7, 7], True)], 4, 0)
    assert (ids, starts, crosses) == ([1, 7, 7], [], [])


def test_split_prompt_keeps_order_and_skips_missing():
    text = "sys A tool B tool A end"
    assert split_prompt(text, ["B", "missing", "A"]) == [
        ("sys A tool ", False),
        ("B", True),
        (" tool ", False),
        ("A", True),
        (" end", False),
    ]


def test_repo_index_matches_numbered_file_slices(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n\n\ndef g():\n    pass\n")
    (tmp_path / "untracked.py").write_text("secret = 1\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    repo = RepoIndex(str(tmp_path))

    assert repo.contains("     5\tdef g():\n     6\t    pass\n")
    assert repo.contains("1→def f():\n2→    return 1")
    assert not repo.contains("def g():\n    return 1")
    assert not repo.contains("secret = 1")
    assert not repo.contains("   \n")
