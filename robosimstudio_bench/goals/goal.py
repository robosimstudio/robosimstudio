# SPDX-License-Identifier: Apache-2.0
"""Turning a checked task document into something that scores an episode.

The last step of the compiler, and the one that makes a BDDL file *runnable*
rather than merely valid. A parsed goal is a tree of conditions; this builds the
matching tree of combinator nodes, and the result satisfies the ``Evaluator``
protocol -- so it drops into ``TaskSpec.evaluator`` exactly where a hand-written
evaluator goes today, and nothing downstream learns that goal trees exist.

**Leaves are built by an injected maker.** Structure and measurement are
separate concerns: mapping ``(once (and A B))`` onto ``Once(AllOf(...))`` is
arithmetic over a tree, while turning ``(>= (winding-turns cable peg0.axis) 0.5)``
into a thing that reads a live scene needs handles, features and world
transforms. Splitting them means the structural half is testable against a
three-line fake, exhaustively, with no scene -- and the structural half is where
a mis-mapped modal silently changes what a benchmark measures.

This lives in the asset library rather than the compiler because building a leaf
needs atoms and handlers, and those are the asset layer's. The toolkit hands up
a parsed document; the asset layer hands back an evaluator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

from robosimstudio_bench.taskdoc.ast import Condition, TaskDoc

from .combinators import After, AllOf, AnyOf, HoldFor, Integral, Leaf, Node, Not, Once, Peak, WorstOf
from .reward import reading_key

__all__ = [
    "Clause",
    "LeafMaker",
    "about_key",
    "build_goal",
    "clause_name",
    "leaves",
    "mask_condition",
    "mask_goal",
    "maskable_objects",
    "outline",
    "placements",
    "reading",
    "stated_gate",
]

LeafMaker = Callable[[Condition], Leaf]
"""Builds the node that actually measures one condition.

Given a leaf condition -- a predicate or a comparison, with its operands already
resolved by the typechecker -- return a :class:`Leaf` that reads a live context
and reports a verdict. What that costs is the asset layer's business; what it
composes into is this module's.
"""

_CONNECTIVES = {"and": AllOf, "any": AnyOf, "or": AnyOf}


def build_goal(condition: Condition, make_leaf: LeafMaker, *, path: str = "goal") -> Node:
    """Build the node tree for one goal.

    Args:
        condition: A checked goal condition.
        make_leaf: Builds the measuring node for each leaf.
        path: Dotted label for this node, so a per-node verdict in ``metrics``
            says *which* clause it came from. That decomposition is what turns a
            zero-reward run from "0.0" into "the wrap held but the cable never
            came to rest".

    Raises:
        ValueError: On a form the combinator set does not cover. Refused rather
            than approximated -- a goal that silently means something adjacent to
            what it says is worse than one that will not build.
    """
    head = condition.modal or condition.head

    if head in _CONNECTIVES:
        children = tuple(build_goal(child, make_leaf, path=f"{path}.{i}") for i, child in enumerate(condition.children))
        if not children:
            raise ValueError(f"{path}: ({head} ...) has no clauses, so it says nothing")
        return _CONNECTIVES[head](of=children, label=path)

    if head == "not":
        if len(condition.children) != 1:
            raise ValueError(f"{path}: (not ...) negates exactly one clause, got {len(condition.children)}")
        return Not(of=build_goal(condition.children[0], make_leaf, path=f"{path}.0"), label=path)

    if head in ("once", "hold-for", "after") and condition.op:
        raise ValueError(
            f"{path}: ({head} ...) is a verdict and takes no comparison around it; "
            f"put the comparison inside, as ({head} ... ({condition.op} MEASURE {condition.value}))"
        )

    if head == "once":
        return Once(of=build_goal(condition.children[0], make_leaf, path=f"{path}.0"), label=path)

    if head == "hold-for":
        seconds = condition.modal_arg
        if not isinstance(seconds, (int, float)):
            raise ValueError(f"{path}: (hold-for ...) needs a duration in seconds, got {seconds!r}")
        return HoldFor(
            of=build_goal(condition.children[0], make_leaf, path=f"{path}.0"),
            seconds=float(seconds),
            label=path,
        )

    if head == "after":
        event = condition.modal_arg
        if not isinstance(event, Condition):
            raise ValueError(f"{path}: (after ...) needs an event clause, got {event!r}")
        return After(
            event=build_goal(event, make_leaf, path=f"{path}.event"),
            of=build_goal(condition.children[0], make_leaf, path=f"{path}.0"),
            label=path,
        )

    if head in ("peak", "integral"):
        # The wrapped clause supplies the quantity; the comparison that was
        # written *around* the modal supplies the threshold. `(<= (peak (t f))
        # 12.0)` parses as a comparison whose head is the modal, so the operator
        # and value are on this condition and the measure is its child.
        inner = condition.children[0]
        node = Peak if head == "peak" else Integral
        if not condition.op:
            raise ValueError(
                f"{path}: ({head} ...) reports a quantity and must be compared -- "
                f"write (<= ({head} ...) VALUE) rather than using it as a verdict"
            )
        return node(
            # Through `reading_key`, not `inner.head`. A head is an *atom* name
            # and the metrics are keyed by what the atom reports: `contact-force`
            # is written with a dash and reported under one with an underscore,
            # `insertion` reports `depth`, and `insertion:lat` reports `lat`.
            # A key that misses raises at the first step (`combinators._reading`),
            # as the reward clause does; reading the head verbatim would have
            # missed on all three.
            of=build_goal(inner, make_leaf, path=f"{path}.0"),
            key=reading_key(inner.head),
            op=condition.op,
            value=float(condition.value or 0.0),
            label=path,
        )

    if head == "worst_of":
        children = tuple(build_goal(child, make_leaf, path=f"{path}.{i}") for i, child in enumerate(condition.children))
        return WorstOf(of=children, label=path)

    if condition.children:
        raise ValueError(
            f"{path}: {head!r} has sub-clauses but is not a combinator. "
            "The combinator set is closed; a new kind of measurement is an atom, not a form."
        )

    leaf = make_leaf(condition)
    if leaf.label == "":
        leaf.label = path
    return leaf


@dataclass(frozen=True, kw_only=True)
class Clause:
    """One node of a goal, as plain data, labelled the way the tree labels it.

    Attributes:
        label: The dotted path :func:`build_goal` gives the matching node --
            ``goal``, ``goal.0``, ``goal.0.2.0.1``. This is the join: a live
            ``EvalResult`` reports every node's verdict in ``metrics`` under
            exactly this key.
        head: The combinator (``and``, ``once``, ``hold-for``) or, on a leaf, the
            measure as the document spells it, hyphens and all.
        op: The comparison, empty on a predicate or a combinator.
        value: What it is compared against.
        modal_arg: The wrapper's own argument -- ``5.0`` for ``hold-for``. The
            event of an ``after`` is *not* here; it is the first child, so a
            reader walking children sees the whole goal.
        operands: The clause's references as written -- ``("towel", "towel.half")``.
            The same strings :attr:`~robosimstudio_bench.goals.combinators.Leaf.about` carries,
            so the two halves of a clause can be matched up without a second
            convention.
        children: Sub-clauses, in document order.
    """

    label: str
    head: str
    op: str = ""
    value: float | None = None
    modal_arg: float | None = None
    operands: tuple[str, ...] = ()
    children: tuple[Clause, ...] = field(default_factory=tuple)

    def walk(self) -> tuple[Clause, ...]:
        """This clause and every clause under it, depth first, document order."""
        return (self, *tuple(node for child in self.children for node in child.walk()))

    @property
    def gate(self) -> str:
        """``"≥ 0.85"``, or ``""`` when the clause states no threshold."""
        return f"{self.op} {self.value:g}" if self.op and self.value is not None else ""


def outline(condition: Condition, *, path: str = "goal") -> Clause:
    """The goal's shape as data, labelled identically to :func:`build_goal`.

    **Why this exists at all.** A built goal tree carries its structure and its
    verdicts and *not its thresholds*: ``0.85`` lives on the
    :class:`~robosimstudio_bench.taskdoc.ast.Condition`, and
    :func:`build_goal` consumes it into a closure rather than storing it. So a
    reader holding only the tree can say a clause failed and cannot say what it
    needed -- which is the difference between "goal.0.1 ✗" and "fold-score 0.81,
    needs ≥ 0.85", and the whole of what makes a goal legible while it runs.

    :func:`stated_gate` answers the same question for one measure by name. This
    answers it for every node at once, keyed the way the verdicts are keyed, which
    is what a tree view needs.

    **It is here, beside the builder, on purpose.** Two recursions over one
    grammar drift the moment a form is added to one of them, and the failure is
    quiet: a mislabelled node reads its sibling's verdict and the page shows a
    confident wrong tick. Keeping them adjacent makes the divergence visible in
    review.

    Args:
        condition: A checked goal condition.
        path: Dotted label for this node. The default matches
            :func:`build_goal`'s.

    Returns:
        The root clause. Unlike :func:`build_goal` this refuses nothing -- a form
        the combinator set does not cover is described as the leaf it looks like,
        because a describer that raises where the builder would is a second
        opinion on validity and this is not the layer that holds one.
    """
    head = condition.modal or condition.head
    kids: tuple[Condition, ...] = condition.children

    if head == "after" and isinstance(condition.modal_arg, Condition):
        # The event is a clause of the goal and gets a label of its own, so it is
        # carried as a child rather than as an argument -- a reader walking
        # children would otherwise never see the thing the goal waits for.
        return Clause(
            label=path,
            head=head,
            operands=_operands(condition),
            children=(
                outline(condition.modal_arg, path=f"{path}.event"),
                *(outline(child, path=f"{path}.0") for child in kids[:1]),
            ),
        )

    return Clause(
        label=path,
        head=head,
        op=condition.op,
        value=condition.value,
        modal_arg=condition.modal_arg if isinstance(condition.modal_arg, (int, float)) else None,
        operands=_operands(condition),
        children=tuple(
            outline(child, path=f"{path}.{index if _numbered(head) else 0}") for index, child in enumerate(kids)
        ),
    )


def _numbered(head: str) -> bool:
    """Whether this form labels its children by position.

    The connectives and ``worst_of`` do; every modal has exactly one body and
    labels it ``.0``. Read off :func:`build_goal`'s own branches -- ``not``,
    ``once``, ``hold-for``, ``after``, ``peak`` and ``integral`` all recurse with
    a literal ``.0`` there.
    """
    return head in _CONNECTIVES or head == "worst_of"


def _operands(condition: Condition) -> tuple[str, ...]:
    """A clause's references as the document wrote them."""
    return tuple(str(ref) for ref in condition.args)


def mask_condition(condition: Condition | None, active: Iterable[str], objects: Iterable[str]) -> Condition | None:
    """The goal with every clause about an object the episode does not run removed.

    What makes an *instance* of a superset score as itself. A superset
    document's goal names every object its model holds -- eight ``In`` clauses
    for eight syringes -- and an episode running four of them would otherwise
    score four of eight forever. The four clauses about the parked syringes are
    dropped before the tree is built, so the conjunction that remains is the
    instance's own and a full sort reads 4/4.

    A clause is *about* an object when any of its references is that object
    (``syringe_pack_3``, ``syringe_pack_3.cap``); references to regions,
    fixtures and features of scenery are not objects and never mask a clause.
    A combinator whose every child is dropped is dropped with them, and a
    modal whose body or event is dropped too -- ``(after (released x) (in x r))``
    about a parked ``x`` says nothing about this episode. Labels are assigned
    when the tree is built, so the surviving clauses are relabelled in their
    new positions and a per-node verdict in ``metrics`` still names the clause
    it came from.

    Args:
        condition: The goal, or any clause of one. ``None`` answers ``None``.
        active: The episode's active set.
        objects: Every **free** object the document declares -- what a
            reference has to be one of to count as parkable
            (:func:`maskable_objects`). A fixture is welded in every world
            whether or not the set names it, so a clause about one is never
            dropped; a name in neither set is not an object and is left alone.

    Returns:
        The masked clause, or ``None`` when nothing of it is about this
        episode. A caller holding ``None`` for a whole goal has an instance
        that runs none of the objects its goal is about, which is a document
        mistake and is refused where the evaluator is built.
    """
    if condition is None:
        return None
    keep = frozenset(str(name) for name in active)
    inactive = frozenset(str(name) for name in objects) - keep
    return _mask(condition, inactive)


def maskable_objects(doc: TaskDoc) -> tuple[str, ...]:
    """The document's free objects: what an active set can leave out.

    ``:objects`` and not ``:fixtures`` -- a fixture lowers to a static handle
    with no free joint, is welded in every world, and is shared by every
    instance whether or not its active set names it. The same rule the
    handle table restricts by (``coord_start < 0`` is never dropped) and the
    parking writes by. Reading every instance as maskable dropped the
    ``encloses`` and ``contact-force`` clauses of a hanging goal, whose hook is
    a fixture, and left an instance that measured only the wrench's speed --
    satisfied before it started.
    """
    return tuple(o.name for o in doc.objects if not o.static)


def mask_goal(doc: TaskDoc, active: Iterable[str] | None) -> Condition | None:
    """:func:`mask_condition` over a document's goal, by the document's own free objects."""
    if active is None:
        return doc.goal
    return mask_condition(doc.goal, active, maskable_objects(doc))


def _mask(condition: Condition, inactive: frozenset[str]) -> Condition | None:
    if any(ref.base in inactive for ref in condition.args):
        return None
    modal_arg = condition.modal_arg
    if isinstance(modal_arg, Condition):
        modal_arg = _mask(modal_arg, inactive)
        if modal_arg is None:
            return None
    if not condition.children:
        return condition if modal_arg is condition.modal_arg else replace(condition, modal_arg=modal_arg)
    children = tuple(kept for kept in (_mask(child, inactive) for child in condition.children) if kept is not None)
    if not children:
        return None
    if children == condition.children and modal_arg is condition.modal_arg:
        return condition
    return replace(condition, children=children, modal_arg=modal_arg)


def clause_name(atom: str, operands: Sequence[str]) -> str:
    """What to call one goal clause on a readout.

    Whichever half carries the information. A clause with two operands is about
    the pair -- ``hammer_0/bin2_hammer_region`` says everything and the atom
    (``containment``) says nothing that is not already obvious. A clause with one
    is about the measure: RJ45's three are all about ``assembly``, so naming them
    by their operand would print the same word three times where ``mate-gap``,
    ``tcp-clearance`` and ``plug-speed`` are the whole content.
    """
    return "/".join(operands) if len(operands) > 1 else atom


def reading(metrics: dict, leaf: Leaf) -> str:
    """One clause's number, or its verdict when it has no number.

    Three shapes, because clauses genuinely differ and there is no one key that
    covers them:

    * a handler writes :func:`about_key` -- the atom and every operand;
    * a node prefixes its own readings with its label, so a clause at ``goal.2``
      reports ``goal.2.plug_speed``;
    * a clause over a *published* state reports its value under the state's
      own name, underscored, and its verdict under :func:`about_key`.

    Tried in that order, ending at the label's own boolean, which every node
    writes and which is therefore the one thing always there. Chasing any single
    one of these was how the peel line came back reading ``tape None`` with
    ``broken_fraction`` sitting in the dict beside it.

    It lives here rather than in the panel that first needed it because it is
    knowledge about which key the *binder* writes, and a second copy beside a
    widget is free to disagree with the binder it is reading.
    """
    atom, operands = leaf.about
    prefixed = next((value for key, value in metrics.items() if key.startswith(f"{leaf.label}.")), None)
    for value in (metrics.get(about_key(atom, operands)), prefixed, metrics.get(atom.replace("-", "_"))):
        if isinstance(value, float):
            return f"{value:.2f}"
        if value is not None and not isinstance(value, bool):
            return str(value)
    verdict = metrics.get(leaf.label, prefixed)
    return "-" if verdict is None else str(verdict)


def leaves(node: Node) -> tuple[Leaf, ...]:
    """Every measuring leaf under ``node``, in the order the document wrote them.

    Order is the content, not an accident of traversal: the tool-sort graph works
    its goals "in the .bddl file's own order", and a policy that took them in a
    different order would be solving a different-looking task on every run.
    """
    if isinstance(node, Leaf):
        return (node,)
    return tuple(leaf for child in node.children() for leaf in leaves(child))


def placements(node: Node, *, atom: str = "containment") -> tuple[tuple[str, str], ...]:
    """The ``(object, region)`` pairs a goal is scored on.

    What a scripted policy actually needs to know, asked of the thing that
    knows it. Before this existed, both the tool-sort routine and its tool
    registry read the pairs out of ``TaskSpec.metadata`` -- a copy, hand-written
    beside the goal that already stated them. A copy can disagree with its
    original, and one that does produces a policy diligently solving a task
    nobody is scoring, with nothing in either file to show for it.

    Args:
        node: A goal tree.
        atom: Which measure counts as a placement. ``containment`` by default,
            which is what ``In`` binds to.

    Returns:
        Pairs in document order. Empty when the goal is not about placing things,
        which is a legitimate answer -- a peel has no placements, and a caller
        that needs some should say so rather than being handed an invented pair.
    """
    return tuple(
        (operands[0], operands[1])
        for leaf in leaves(node)
        for name, operands in (leaf.about,)
        if name == atom and len(operands) >= 2
    )


def stated_gate(condition: Condition | None, head: str) -> float | None:
    """The threshold a document states for one measure, or ``None``.

    What a solution needs when it wants to stop where the *task* says rather than
    where the asset would. The two are usually the same number and the asset's is
    the honest default -- it is a fact about the connector, not about the job --
    but a task is entitled to a stricter line for its own purposes, and a graph
    reading the asset's constant would go on reporting `inserted: true` while the
    goal said false.

    That is not hypothetical even today, only harmless: the USB-C document writes
    its lateral guard to ten significant figures where the asset computes it to
    full precision, so the two differ by 2.5e-11 and have been separately
    maintained all along.

    Args:
        condition: A document's goal, or any clause of one. ``None`` is allowed
            and answers ``None`` -- not every task is defined by a document.
        head: The measure's name as the document spells it, hyphens and all --
            ``insertion``, ``insertion:lat``, ``mate-gap``.

    Returns:
        The first threshold stated for that measure, walking the clause tree in
        document order, or ``None`` when the document does not gate it. First
        rather than all, because a measure gated twice in one goal is a document
        that means two different things by one word, which the typechecker is the
        place to complain about.
    """
    if condition is None:
        return None
    if condition.head == head and condition.value is not None:
        return float(condition.value)
    for child in condition.children:
        found = stated_gate(child, head)
        if found is not None:
            return found
    return None


def about_key(atom: str, operands: Sequence[str]) -> str:
    """The metric key a bound clause reports its verdict under.

    ``containment/hammer_0/bin2_hammer_region``: the atom and every operand. One
    function rather than two format strings, because the binder writes this key
    and policies read it, and the failure when those drift is silent.

    A leaf carries both parts, so a caller holding one writes
    ``about_key(*leaf.about)``. It cost a
    graph rollout to learn that: the shipped tool registry looked for
    ``in/hammer_0`` -- the *previous* evaluator's key -- found nothing, concluded
    no goal was ever satisfied, and re-attempted the same tool until the episode
    truncated. Nothing raised. The run simply read as a policy that could not
    sort one hammer.
    """
    return "/".join((atom, *operands))
