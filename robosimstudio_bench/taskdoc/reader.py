# SPDX-License-Identifier: Apache-2.0
"""Reading s-expressions, with errors that say where.

A hundred lines, no dependency. The ``bddl`` package on PyPI is pinned off in
this repo (``sys_platform == 'never'``, alongside robosuite) and it carries the
BEHAVIOR ontology this stack has no use for, so parsing the dialect directly is
both cheaper and more honest than depending on a package for a reader.

**Errors carry a line and a column, always.** That is not politeness. Task files
are written by hand and emitted by models, and a model that gets a paren wrong
has to be told *where* to be able to fix it -- "unexpected )" with no position is
a message that turns a one-token error into a re-read of the whole file. Every
raise below names a position.

The BDDL-specific things the reader must get right:

* ``;`` runs to end of line, and the vendored files use comments heavily -- the
  provenance in a reconstructed file is all comments.
* Tokens are broad. ``:ranges``, ``bin2_hammer_region``, ``-0.195``, ``?t`` and
  ``>=`` are all single atoms, so the only delimiters are parens, whitespace and
  the comment character. Numbers are converted; everything else stays a string,
  because deciding what a symbol *means* is the parser's job and not the
  reader's.
* ``object.feature(NUMBER)`` -- ``cable.point(0.63)``, the paren hard against a
  dotted name -- is one atom, a parameterised reference, not a name followed
  by a form.
* ``"..."`` is one string atom. Quotes stripped, ``;`` and parens literal
  inside, and it may span lines -- whitespace runs collapse to one space, since
  the only strings in the dialect are prose and references. A quoted ``"5"``
  stays a string.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["ParseError", "Sexp", "parse_sexp", "read_sexp"]

Sexp = str | float | int | list["Sexp"]
"""An s-expression: an atom, a number, or a list of them."""

DELIMITERS = "()"
COMMENT = ";"
QUOTE = '"'


class ParseError(ValueError):
    """A malformed s-expression, with where it went wrong.

    Attributes:
        line: 1-based line.
        column: 1-based column.
        where: The file, when there was one.
    """

    def __init__(self, message: str, *, line: int, column: int, where: str = "") -> None:
        location = f"{where}:{line}:{column}" if where else f"line {line}, column {column}"
        super().__init__(f"{location}: {message}")
        self.line = line
        self.column = column
        self.where = where


_REF_ARGUMENT = re.compile(r"\(\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\)")
"""``(0.63)`` written hard against a reference: the argument of ``cable.point(0.63)``."""


def _is_dotted_name(token: str) -> bool:
    """Whether ``token`` is ``object.feature`` -- dotted, and not a number such as ``0.5``."""
    return "." in token and isinstance(_atom(token), str) and token.rsplit(".", 1)[1][:1].isalpha()


def _atom(token: str) -> Sexp:
    """A token as a number when it is one, and a string otherwise.

    ``-0.195`` is a number; ``-`` is the typing separator and stays a string;
    ``bin2_hammer_region`` stays a string. Deliberately narrow: a reader that
    guesses at meaning makes the parser's job harder, not easier.
    """
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_sexp(text: str, *, where: str = "") -> list[Sexp]:
    """Parse every top-level form in ``text``.

    Returns a list because a file may hold more than one -- a domain and a
    problem can share a file, and refusing that would be a restriction the format
    does not have.

    Raises:
        ParseError: On an unclosed form, a stray close paren, or a bare atom at
            top level. Each names its position.
    """
    forms: list[Sexp] = []
    stack: list[list[Sexp]] = []
    # Where the currently-open forms started, so an unclosed paren can point at
    # the one that opened rather than at the end of the file -- which is the
    # difference between a useful message and "unexpected EOF".
    opened: list[tuple[int, int]] = []

    line = column = 1
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == COMMENT:
            while index < length and text[index] != "\n":
                index += 1
            continue

        if char == "\n":
            line += 1
            column = 1
            index += 1
            continue

        if char.isspace():
            column += 1
            index += 1
            continue

        if char == QUOTE:
            start_line, start_column = line, column
            index += 1
            column += 1
            chars: list[str] = []
            while index < length and text[index] != QUOTE:
                if text[index] == "\n":
                    line += 1
                    column = 1
                else:
                    column += 1
                chars.append(text[index])
                index += 1
            if index >= length:
                raise ParseError(
                    "unclosed '\"' -- this string is never closed", line=start_line, column=start_column, where=where
                )
            index += 1
            column += 1
            if not stack:
                raise ParseError(
                    "bare string outside any form; BDDL is a list of forms",
                    line=start_line,
                    column=start_column,
                    where=where,
                )
            stack[-1].append(" ".join("".join(chars).split()))
            continue

        if char == "(":
            stack.append([])
            opened.append((line, column))
            column += 1
            index += 1
            continue

        if char == ")":
            if not stack:
                raise ParseError("stray ')' -- nothing is open here", line=line, column=column, where=where)
            done = stack.pop()
            opened.pop()
            if stack:
                stack[-1].append(done)
            else:
                forms.append(done)
            column += 1
            index += 1
            continue

        start_column = column
        token_start = index
        while (
            index < length
            and not text[index].isspace()
            and text[index] not in DELIMITERS
            and text[index] not in (COMMENT, QUOTE)
        ):
            index += 1
            column += 1
        token = text[token_start:index]
        argument = _REF_ARGUMENT.match(text, index) if _is_dotted_name(token) else None
        if argument is not None:
            # `cable.point(0.63)`: a parameterised reference, one atom. The
            # paren hard against a dotted name is not a form -- a form is
            # separated from what precedes it -- and splitting it left
            # `parse_ref` a reference with no argument and a stray `(0.63)`.
            token = f"{token}({argument.group(1)})"
            column += argument.end() - index
            index = argument.end()

        if not stack:
            raise ParseError(
                f"bare atom {token!r} outside any form; BDDL is a list of forms",
                line=line,
                column=start_column,
                where=where,
            )
        stack[-1].append(_atom(token))

    if stack:
        # The *innermost* form still open, not the outermost: it is nearest the
        # paren that was actually forgotten, where the outermost is only ever
        # wherever the file happened to start.
        open_line, open_column = opened[-1]
        raise ParseError(
            f"unclosed '(' -- this form is never closed ({len(stack)} still open at end of input)",
            line=open_line,
            column=open_column,
            where=where,
        )
    return forms


def read_sexp(path: str | Path) -> list[Sexp]:
    """Parse a file, naming it in any error."""
    target = Path(path)
    return parse_sexp(target.read_text(), where=str(target))
