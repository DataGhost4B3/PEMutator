"""
pemutator.core.mutator
----------------------
LIEF-backed, functionality-preserving PE mutation functions.

Each function accepts a file *path* (str) — not raw bytes — because
LIEF's parser reads the file directly.  Each function returns the
mutated binary as raw ``bytes``.

These are the same four mutations implemented across the try*.py
experiments, now collected into a single, documented module.

Mutation catalogue
------------------
append_bytes(path, n)   — Append n null bytes to the PE overlay
add_import(path)        — Add a dummy DLL entry to the import table
pad_header(path)        — Increase SizeOfHeaders by 512 bytes
rename_section(path)    — Rename the first section to ".abcd"

The ``MUTATIONS`` dict maps short string keys to callables that
accept only a path, making it easy to iterate over all mutations
in a loop (see analysis.sweep).

Example
-------
    from pemutator.core.mutator import append_bytes, MUTATIONS

    raw = append_bytes("sample.exe", 5000)
    # raw is bytes — pass to extractor.extract(raw)

    for name, fn in MUTATIONS.items():
        mutated = fn("sample.exe")
"""

from __future__ import annotations

import lief  # type: ignore[import]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build(binary: lief.PE.Binary) -> bytes:
    """Build a LIEF binary to bytes using PE.Builder."""
    builder = lief.PE.Builder(binary)
    builder.build()
    return bytes(builder.get_build())


# ---------------------------------------------------------------------------
# Public mutation functions
# ---------------------------------------------------------------------------

def append_bytes(path: str, n: int = 100) -> bytes:
    """
    Append ``n`` null bytes to the overlay (end) of the PE file.

    This is the most direct size-inflating mutation.  As shown in the
    size-sweep experiments (try7.py / try8.py), it causes feature[0]
    to increase linearly while feature[257] steps discretely at
    alignment boundaries, leading to piecewise-constant prediction
    changes.

    Parameters
    ----------
    path : str
        Path to the PE file to mutate.
    n : int
        Number of null bytes to append (default 100).

    Returns
    -------
    bytes
        Mutated PE file as raw bytes.
    """
    binary = lief.parse(path)
    if binary is None:
        raise ValueError(f"LIEF could not parse PE file: {path!r}")
    data = bytearray(_build(binary))
    data += b"\x00" * n
    return bytes(data)


def add_import(path: str, dll: str = "kernel32.dll") -> bytes:
    """
    Add a dummy DLL entry to the import address table.

    This is the sole manipulation explored by Hu & Tan (MalGAN).
    In the try*.py experiments it was observed to have relatively
    little effect on prediction, suggesting the models are less
    sensitive to import-table changes than to size-related features.

    Parameters
    ----------
    path : str
        Path to the PE file to mutate.
    dll : str
        DLL name to add (default "kernel32.dll").

    Returns
    -------
    bytes
        Mutated PE file as raw bytes.
    """
    binary = lief.parse(path)
    if binary is None:
        raise ValueError(f"LIEF could not parse PE file: {path!r}")
    binary.add_library(dll)
    return _build(binary)


def pad_header(path: str, padding: int = 512) -> bytes:
    """
    Increase ``SizeOfHeaders`` in the optional header by ``padding`` bytes.

    Parameters
    ----------
    path : str
        Path to the PE file.
    padding : int
        Number of bytes to add to SizeOfHeaders (default 512).

    Returns
    -------
    bytes
        Mutated PE file as raw bytes.
    """
    binary = lief.parse(path)
    if binary is None:
        raise ValueError(f"LIEF could not parse PE file: {path!r}")
    binary.optional_header.sizeof_headers += padding
    return _build(binary)


def rename_section(path: str, new_name: str = ".abcd") -> bytes:
    """
    Rename the first PE section to ``new_name``.

    If the binary has no sections the file is returned unmodified.

    Parameters
    ----------
    path : str
        Path to the PE file.
    new_name : str
        New section name (max 8 characters; default ".abcd").

    Returns
    -------
    bytes
        Mutated PE file as raw bytes.
    """
    binary = lief.parse(path)
    if binary is None:
        raise ValueError(f"LIEF could not parse PE file: {path!r}")
    if binary.sections:
        binary.sections[0].name = new_name[:8]
    return _build(binary)


# ---------------------------------------------------------------------------
# Mutation registry
# ---------------------------------------------------------------------------

MUTATIONS: dict[str, callable] = {
    "append":  lambda p: append_bytes(p, 50_000),
    "import":  add_import,
    "header":  pad_header,
    "section": rename_section,
}
"""
Pre-configured mutation callables.  Each value is ``fn(path: str) -> bytes``.
The ``append`` variant uses 50 000 bytes to match the try*.py experiments.

Keys
----
"append"   append_bytes with n=50 000
"import"   add_import
"header"   pad_header
"section"  rename_section
"""
