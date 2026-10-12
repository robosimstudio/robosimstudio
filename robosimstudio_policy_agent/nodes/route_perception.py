"""Which perception body runs: the privileged oracle, or the camera.

A router rather than two graph directories. The alternative -- shipping the whole
workflow twice, once per body, held together by SHA256 byte-identity tests -- was
tried elsewhere in this corpus and costs two tests per suite to enforce something
one edge states. Here the two bodies are sibling subgraphs declaring the same
inputs, the same output keys and the same exits, so only one of them ever writes
into the cross-subgraph namespace and every downstream reference is unchanged.

The value arrives as a top-level workflow input (``--perception``), not as
anything the runtime interprets: a graph that declares no such router simply
never reads it.
"""

from typing import TypedDict

from gap import NodeContext

BODIES = ("oracle", "vision")
"""The two bodies of the perception slot. Exactly these, and refusing anything
else -- a mistyped body that silently fell through to the oracle would report a
vision run that never happened, which is the one failure this pair exists to
make impossible."""


class Output(TypedDict):
    route: str


def run(ctx: NodeContext, perception: str = "oracle") -> Output:
    del ctx
    body = str(perception or "oracle").strip().lower()
    if body not in BODIES:
        raise ValueError(f"unknown perception body {perception!r}; have {list(BODIES)}")
    return {"route": body}
