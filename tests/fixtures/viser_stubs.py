# SPDX-License-Identifier: Apache-2.0
"""Viser stand-ins: the studio's browser surface, without a browser.

The studio is library code and the interface this stack is used through, so its
panels, pickers, frustums and take-recorder are worth testing -- and none of that
needs a browser, a websocket or a human. These stubs stand in for viser's GUI and
scene handles: a widget is a value, a disabled flag and callbacks a test may
fire; a scene is a list of what was added to it.

Promoted out of a physics test module, where eight of them sat in the middle of two
and a half thousand lines of physics tests. Nothing about them is
physics-specific and nothing about them needs a GPU -- which is exactly why they
belong in the fixtures package: the studio tests that use them can then live
beside the other studio tests rather than beside the tape scene.

The pattern to preserve when extending these: a stub records what it was asked to
do and does nothing else. A stub that starts *behaving* -- laying out widgets,
computing what a real panel would compute -- is a second implementation of the
thing under test, and it will agree with the real one right up until it matters.
"""

from __future__ import annotations


class _StubWidget:
    """A viser widget: a value, a disabled flag, and callbacks a test may fire."""

    def __init__(self, value=None) -> None:
        self.value = value
        self.disabled = False
        self.removed = False
        self.callbacks: list = []

    def on_update(self, fn):
        self.callbacks.append(fn)
        return fn

    def on_click(self, fn):
        self.callbacks.append(fn)
        return fn

    def fire(self, value=None) -> None:
        """What the browser does: set the value, then tell the server."""
        if value is not None:
            self.value = value
        for fn in list(self.callbacks):
            fn(self)

    def remove(self) -> None:
        self.removed = True


class _StubImage(_StubWidget):
    def __init__(self, image, label) -> None:
        super().__init__()
        self.image = image
        self.label = label


class _StubMarkdown:
    """A markdown block: its content, which a panel rewrites on refresh, and the
    folder it was written into -- which is how a test asks what a section shows
    before anything is unrolled."""

    def __init__(self, content: str, parent=None) -> None:
        self.content = content
        self.parent = parent
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _StubFolder:
    """A container, and where it was opened.

    ``parent`` is what makes nesting checkable: a page that unrolls twice to
    reach one control is a folder whose parent is a folder, and that is a claim
    a test can make without a browser. Entering pushes onto the gui's open
    stack, which is how the *next* folder learns who its parent is -- so the
    bookkeeping is on the type, where ``with`` will actually find it.
    """

    def __init__(self, label=None, parent=None, gui=None) -> None:
        self.label = label
        self.parent = parent
        self.removed = False
        self._gui = gui

    def __enter__(self):
        if self._gui is not None:
            self._gui._open.append(self)
        return self

    def __exit__(self, *exc) -> bool:
        if self._gui is not None and self._gui._open and self._gui._open[-1] is self:
            self._gui._open.pop()
        return False

    def remove(self) -> None:
        self.removed = True


class _StubGui:
    """Enough of ``server.gui`` for a panel to build and publish into."""

    def __init__(self) -> None:
        self.images: list[_StubImage] = []
        self.widgets: list[tuple[str, _StubWidget]] = []
        self.folders: list[_StubFolder] = []
        self.markdown: list[_StubMarkdown] = []
        self._open: list[_StubFolder] = []
        """The folders a build is currently inside, innermost last."""

    def _widget(self, label, value, *, disabled: bool = False, **_ignored) -> _StubWidget:
        """One widget under ``label``. ``disabled`` is kept, as viser keeps it:
        a panel that builds a switch greyed out is making a claim a test may
        read, and a stub that dropped the keyword answered "enabled" for every
        widget whatever the panel had said."""
        handle = _StubWidget(value)
        handle.disabled = bool(disabled)
        self.widgets.append((str(label), handle))
        return handle

    def find(self, label: str) -> _StubWidget:
        """The widget a panel built under this label, for a test to drive."""
        return next(w for name, w in self.widgets if name == label)

    def add_folder(self, label=None, **kwargs):
        folder = _StubFolder(label, self._open[-1] if self._open else None, gui=self)
        self.folders.append(folder)
        return folder

    def add_markdown(self, content="", **kwargs):
        handle = _StubMarkdown(str(content), self._open[-1] if self._open else None)
        self.markdown.append(handle)
        return handle

    def add_checkbox(self, label, initial_value=False, **kwargs):
        return self._widget(label, bool(initial_value), **kwargs)

    def add_dropdown(self, label, options=(), initial_value=None, **kwargs):
        return self._widget(label, initial_value if initial_value is not None else options[0], **kwargs)

    def add_button_group(self, label, options=(), **kwargs):
        """A row of buttons. ``fire("Close")`` is what pressing one of them does.

        Its handle carries the pressed option as ``value``, which is what the
        panel's callback reads -- so a test presses a button by naming it, the
        way the browser does, rather than by setting the state behind it.
        """
        return self._widget(label, options[0] if options else None, **kwargs)

    def add_number(self, label, initial_value=0, **kwargs):
        return self._widget(label, initial_value, **kwargs)

    def add_slider(self, label, min=0.0, max=1.0, step=0.1, initial_value=0.0, **kwargs):
        return self._widget(label, initial_value, **kwargs)

    def add_text(self, label, initial_value="", **kwargs):
        return self._widget(label, initial_value, **kwargs)

    def add_button(self, label, **kwargs):
        return self._widget(label, label, **kwargs)

    def add_image(self, image, label=None, **kwargs):
        handle = _StubImage(image, label)
        self.images.append(handle)
        return handle


class _StubNode(_StubWidget):
    """A scene node handle: a pose, a visibility, and a way to give it back."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.position = (0.0, 0.0, 0.0)
        self.wxyz = (1.0, 0.0, 0.0, 0.0)
        self.visible = False


class _StubFrustum(_StubNode):
    """A camera frustum handle: a pose, an image, a colour and a click callback."""

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name)
        self.visible = True
        self.fov = kwargs.get("fov")
        self.aspect = kwargs.get("aspect")
        self.scale = kwargs.get("scale")
        self.color = kwargs.get("color")
        self.image = kwargs.get("image")
        self.position = kwargs.get("position", (0.0, 0.0, 0.0))
        self.wxyz = kwargs.get("wxyz", (1.0, 0.0, 0.0, 0.0))
        self.clicked = []

    def on_click(self, fn):
        self.clicked.append(fn)
        return fn


class _StubScene:
    """Enough of ``server.scene`` for a panel's gizmo, handles and frustums."""

    def __init__(self) -> None:
        self.nodes: list[_StubNode] = []
        # viser keeps one of these and it is the only way to ask what is in the
        # scene by name, which the viewport panel needs to hide the model and
        # keep the furniture. Stubbed under its real name on purpose: a rename
        # upstream should fail a test here rather than quietly draw nothing.
        self._handle_from_node_name: dict[str, _StubNode] = {}

    def _node(self, name: str, visible: bool = False) -> _StubNode:
        node = _StubNode(name)
        node.visible = bool(visible)
        self.nodes.append(node)
        self._handle_from_node_name[name] = node
        return node

    def set_background_image(self, image, **kwargs):
        self.background = image

    def add_camera_frustum(self, name, **kwargs):
        node = _StubFrustum(name, **kwargs)
        self.nodes.append(node)
        self._handle_from_node_name[name] = node
        return node

    def add_transform_controls(self, name, **kwargs):
        return self._node(name, kwargs.get("visible", True))

    def add_mesh_simple(self, name, **kwargs):
        return self._node(name, kwargs.get("visible", False))

    def add_icosphere(self, name, **kwargs):
        return self._node(name, kwargs.get("visible", True))

    def on_click(self, *args, **kwargs):
        def register(fn):
            self.click_callbacks.append(fn)
            return fn

        return register

    def remove_click_callback(self, callback="all"):
        if callback == "all":
            self.click_callbacks.clear()
        else:
            self.click_callbacks[:] = [fn for fn in self.click_callbacks if fn != callback]

    @property
    def click_callbacks(self) -> list:
        """What ``on_click`` registered and ``remove_click_callback`` has not removed: viser keeps a list too."""
        if "_clicks" not in self.__dict__:
            self._clicks = []
        return self._clicks


class _StubSession:
    def __init__(self, env, gui) -> None:
        from robosimstudio.core.profile import NULL

        self.env = env
        # ``get_clients`` because a panel that renders the browser's own camera
        # has to ask who is looking; an empty dict is a page nobody has opened.
        self.server = type(
            "_Server", (), {"gui": gui, "scene": _StubScene(), "get_clients": staticmethod(lambda: {})}
        )()
        self.status = ""
        self.obs: dict = {}
        self.paused = False
        self.arms: dict = {}
        # The session's measurement surface, which panels now read: the
        # disabled profiler, and the draw cadence the frustum pass paces by.
        self.profile = NULL
        self.render_every = 4


__all__ = [
    "_StubFolder",
    "_StubFrustum",
    "_StubGui",
    "_StubImage",
    "_StubMarkdown",
    "_StubNode",
    "_StubScene",
    "_StubSession",
    "_StubWidget",
]
