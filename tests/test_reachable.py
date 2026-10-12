"""No statement in this package may sit after the one that leaves its block.

Ruff has no rule for this, and the gap is not theoretical: inserting a branch
into ``StudioSession.tick`` at one indent too deep put thirty lines -- the
Reset button, the scenery request, the cloth look -- *after* a return in the
same block. Every one of them stopped running, in both modes, and nothing
failed: the page still stepped, the tests still passed, and the only symptom
was a button that quietly did nothing.

That is the shape of bug worth a structural test rather than a behavioural one.
A behavioural test would have to know which handler to press; this knows only
that code which cannot execute is a mistake, wherever it appears.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

TIER = "fast"
"""No GPU. Gates every commit."""

ROOT = pathlib.Path(__file__).resolve().parent.parent / "robosimstudio"

TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _dead(tree: ast.AST) -> list[tuple[int, str]]:
    """Every statement that follows a terminal one in the same block."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, statement in enumerate(block[:-1]):
                if isinstance(statement, TERMINAL):
                    after = block[i + 1]
                    found.append((after.lineno, ast.unparse(after).splitlines()[0][:60]))
                    break
    return found


class TestNothingUnreachable(unittest.TestCase):
    def test_no_statement_follows_a_return_raise_break_or_continue(self):
        offenders: list[str] = []
        for path in sorted(ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for line, text in _dead(tree):
                offenders.append(f"{path.relative_to(ROOT.parent)}:{line}: unreachable -- {text}")
        self.assertEqual(offenders, [], "unreachable code:\n" + "\n".join(offenders))

    def test_the_check_finds_what_it_is_for(self):
        """The bug that motivated this, in miniature -- so a broken check fails."""
        source = "def f(x):\n    if x:\n        return 1\n        return 2\n"
        self.assertEqual([line for line, _ in _dead(ast.parse(source))], [4])

    def test_a_return_at_the_end_of_a_block_is_fine(self):
        source = "def f(x):\n    if x:\n        return 1\n    return 2\n"
        self.assertEqual(_dead(ast.parse(source)), [])


if __name__ == "__main__":
    unittest.main()
