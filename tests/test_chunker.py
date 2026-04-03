from __future__ import annotations

from pathlib import Path

import pytest
from token_reducer.chunker import (
    char_ngrams,
    chunk_code,
    chunk_text,
    clean_text,
    collect_input_files,
    estimate_tokens,
    extract_function_calls,
    extract_imports,
    is_code_file,
    is_minified,
    is_text_file,
    normalize_chunking,
    read_text_file,
    should_skip_file,
    tokenize,
)
from token_reducer.config import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS


class TestEstimateTokens:
    def test_basic(self) -> None:
        # 2 words * 1.3 = 2.6 → int = 2 → max(1, 2) = 2
        assert estimate_tokens("hello world") == 2

    def test_empty_returns_one(self) -> None:
        assert estimate_tokens("") == 1

    def test_single_word(self) -> None:
        # int(1 * 1.3) = 1 → max(1, 1) = 1
        assert estimate_tokens("hello") == 1

    def test_many_words(self) -> None:
        text = " ".join(["word"] * 100)
        assert estimate_tokens(text) == 130  # int(100 * 1.3) = 130


class TestTokenize:
    def test_basic(self) -> None:
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_numbers_included(self) -> None:
        assert tokenize("foo123 bar") == ["foo123", "bar"]

    def test_empty(self) -> None:
        assert tokenize("") == []

    def test_underscores_kept(self) -> None:
        assert tokenize("foo_bar") == ["foo_bar"]


class TestCharNgrams:
    def test_basic_trigrams(self) -> None:
        grams = char_ngrams("abc", n=3)
        # padded = "^abc$" → "^ab", "abc", "bc$"
        assert grams == ["^ab", "abc", "bc$"]

    def test_short_token_returns_full_padded(self) -> None:
        # padded = "^a$", len=3, range(0, 3-3+1)=range(0,1) → ["^a$"]
        grams = char_ngrams("a", n=3)
        assert grams == ["^a$"]

    def test_empty_token(self) -> None:
        # padded = "^$", len=2 < n=3 → returns ["^$"]
        grams = char_ngrams("", n=3)
        assert grams == ["^$"]


class TestCleanText:
    def test_removes_blank_lines(self) -> None:
        result = clean_text("hello\n\n\nworld")
        assert "hello" in result
        assert "world" in result
        assert "\n\n" not in result

    def test_removes_copyright_lines(self) -> None:
        text = "good content\nCopyright (c) 2024 Acme Corp\nmore content"
        result = clean_text(text)
        assert "Copyright" not in result
        assert "good content" in result

    def test_removes_all_rights_reserved(self) -> None:
        text = "useful text\nAll rights reserved.\nmore useful text"
        result = clean_text(text)
        assert "All rights reserved" not in result

    def test_deduplicates_lines(self) -> None:
        text = "repeated line\nrepeated line\nrepeated line"
        result = clean_text(text)
        assert result.count("repeated line") == 1

    def test_removes_lines_shorter_than_two(self) -> None:
        text = "a\nhello world\nb\nmore content"
        result = clean_text(text)
        lines = result.splitlines()
        assert all(len(line) >= 2 for line in lines)


class TestChunkText:
    def test_empty_returns_empty(self) -> None:
        assert chunk_text("", 100, 0) == []

    def test_single_line_fits_in_one_chunk(self) -> None:
        chunks = chunk_text("hello world", chunk_size_words=100, overlap_words=0)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_produces_multiple_chunks(self) -> None:
        # 10 lines × 50 words each = 500 words; chunk_size=100 → ~5 chunks
        lines = [" ".join(["word"] * 50) for _ in range(10)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_size_words=100, overlap_words=0)
        assert len(chunks) >= 3

    def test_no_content_lost(self) -> None:
        lines = [f"line_{i}_content" for i in range(20)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_size_words=10, overlap_words=0)
        combined = "\n".join(chunks)
        for i in range(20):
            assert f"line_{i}_content" in combined

    def test_chunk_word_count_bounded(self) -> None:
        # Each line = 10 words; chunk_size=20 → 2 lines per chunk = 20 words
        lines = [" ".join([f"w{j}"] * 10) for j in range(20)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_size_words=20, overlap_words=0)
        for chunk in chunks:
            assert len(chunk.split()) <= 25  # small tolerance for boundary lines


class TestIsCodeFile:
    def test_python_is_code(self) -> None:
        assert is_code_file("script.py") is True

    def test_typescript_is_code(self) -> None:
        assert is_code_file("app.ts") is True

    def test_markdown_is_not_code(self) -> None:
        assert is_code_file("readme.md") is False

    def test_text_is_not_code(self) -> None:
        assert is_code_file("notes.txt") is False


class TestIsTextFile:
    def test_markdown(self) -> None:
        assert is_text_file(Path("readme.md")) is True

    def test_python(self) -> None:
        assert is_text_file(Path("script.py")) is True

    def test_png_is_not_text(self) -> None:
        assert is_text_file(Path("image.png")) is False


class TestShouldSkipFile:
    def test_minified_js(self, tmp_path: Path) -> None:
        f = tmp_path / "app.min.js"
        f.write_text("x")
        assert should_skip_file(f) is True

    def test_bundle_js(self, tmp_path: Path) -> None:
        f = tmp_path / "app.bundle.js"
        f.write_text("x")
        assert should_skip_file(f) is True

    def test_source_map(self, tmp_path: Path) -> None:
        f = tmp_path / "app.js.map"
        f.write_text("x")
        assert should_skip_file(f) is True

    def test_svg(self, tmp_path: Path) -> None:
        f = tmp_path / "icon.svg"
        f.write_text("<svg/>")
        assert should_skip_file(f) is True

    def test_lockfile(self, tmp_path: Path) -> None:
        f = tmp_path / "package-lock.json"
        f.write_text("{}")
        assert should_skip_file(f) is True

    def test_yarn_lock(self, tmp_path: Path) -> None:
        f = tmp_path / "yarn.lock"
        f.write_text("")
        assert should_skip_file(f) is True

    def test_normal_python_file(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("print('hi')")
        assert should_skip_file(f) is False

    def test_oversized_file(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (512 * 1024 + 1))
        assert should_skip_file(f) is True


class TestIsMinified:
    def test_normal_text_not_minified(self) -> None:
        text = "This is normal text\nWith multiple lines\nAll short"
        assert is_minified(text) is False

    def test_very_long_line_detected(self) -> None:
        text = "x" * 501
        assert is_minified(text) is True

    def test_long_line_beyond_sample_not_detected(self) -> None:
        # Long line beyond default sample_lines=10 is not checked
        short_lines = ["short line"] * 14
        text = "\n".join(short_lines) + "\n" + ("x" * 501)
        assert is_minified(text) is False


class TestNormalizeChunking:
    def test_normal_values_unchanged(self) -> None:
        chunk, overlap = normalize_chunking(220, 40)
        assert chunk == 220
        assert overlap == 40

    def test_chunk_below_min_raised(self) -> None:
        chunk, _ = normalize_chunking(50, 10)
        assert chunk == MIN_CHUNK_WORDS

    def test_chunk_above_max_capped(self) -> None:
        chunk, _ = normalize_chunking(1000, 10)
        assert chunk == MAX_CHUNK_WORDS

    def test_overlap_equal_to_chunk_adjusted(self) -> None:
        _, overlap = normalize_chunking(220, 220)
        assert overlap < 220

    def test_overlap_exceeds_45pct_capped(self) -> None:
        # max_overlap = max(20, int(220 * 0.45)) = 99
        _, overlap = normalize_chunking(220, 150)
        assert overlap <= int(220 * 0.45)


class TestChunkCode:
    def test_python_content_preserved(self) -> None:
        code = "import os\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        chunks = chunk_code(code, "test.py", chunk_size_words=MIN_CHUNK_WORDS)
        combined = "\n".join(chunks)
        assert "def foo" in combined
        assert "def bar" in combined

    def test_unknown_extension_falls_back_to_plain(self) -> None:
        text = "some text content here\nmore lines\n"
        chunks = chunk_code(text, "test.xyz", chunk_size_words=100)
        assert len(chunks) >= 1

    def test_empty_code_returns_empty(self) -> None:
        assert chunk_code("", "test.py", chunk_size_words=100) == []

    def test_large_python_produces_multiple_chunks(self) -> None:
        # Build code that definitely exceeds MIN_CHUNK_WORDS*2 to force multiple chunks
        funcs = "\n\n".join(
            f"def func_{i}():\n    " + "x = 1\n    " * 20 + "return x" for i in range(20)
        )
        chunks = chunk_code(funcs, "big.py", chunk_size_words=MIN_CHUNK_WORDS)
        assert len(chunks) >= 2


class TestExtractImports:
    def test_python_from_import(self) -> None:
        code = "from os.path import join\nimport sys"
        imports = extract_imports(code, "test.py")
        assert "os.path" in imports
        assert "sys" in imports

    def test_js_es_module(self) -> None:
        code = "import React from 'react';"
        imports = extract_imports(code, "test.js")
        assert "react" in imports

    def test_js_require(self) -> None:
        code = "const fs = require('fs');"
        imports = extract_imports(code, "test.js")
        assert "fs" in imports

    def test_unknown_extension_returns_empty(self) -> None:
        assert extract_imports("some text", "test.xyz") == []


class TestExtractFunctionCalls:
    def test_basic_calls_detected(self) -> None:
        calls = extract_function_calls("foo(x) + bar(y)")
        assert "foo" in calls
        assert "bar" in calls

    def test_builtins_filtered(self) -> None:
        calls = extract_function_calls("print(z) + len(a) + str(b) + foo(w)")
        assert "print" not in calls
        assert "len" not in calls
        assert "str" not in calls
        assert "foo" in calls

    def test_keywords_filtered(self) -> None:
        calls = extract_function_calls("if(x) for(y) foo(w)")
        assert "if" not in calls
        assert "for" not in calls

    def test_private_names_filtered(self) -> None:
        calls = extract_function_calls("_private(x) public_fn(y)")
        assert "_private" not in calls
        assert "public_fn" in calls

    def test_deduplicates_results(self) -> None:
        calls = extract_function_calls("foo(1) + foo(2) + foo(3)")
        assert calls.count("foo") == 1


class TestReadTextFile:
    def test_reads_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_text_file(f) == "hello world"

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert read_text_file(tmp_path / "nonexistent.txt") is None


class TestCollectInputFiles:
    def test_collects_text_files_from_directory(self, tmp_path: Path) -> None:
        (tmp_path / "script.py").write_text("pass")
        (tmp_path / "notes.txt").write_text("notes")
        files = collect_input_files([str(tmp_path)])
        names = {f.name for f in files}
        assert "script.py" in names
        assert "notes.txt" in names

    def test_skips_ignored_dirs(self, tmp_path: Path) -> None:
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.js").write_text("module = {}")
        (tmp_path / "main.py").write_text("pass")
        files = collect_input_files([str(tmp_path)])
        names = {f.name for f in files}
        assert "package.js" not in names
        assert "main.py" in names

    def test_warns_on_nonexistent_path(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        collect_input_files([str(tmp_path / "nonexistent")])
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_single_file_input(self, tmp_path: Path) -> None:
        f = tmp_path / "solo.py"
        f.write_text("x = 1")
        files = collect_input_files([str(f)])
        assert len(files) == 1
        assert files[0].name == "solo.py"
