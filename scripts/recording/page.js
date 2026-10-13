// SPDX-License-Identifier: Apache-2.0
//
// window.__rss: what the recorder asks a studio page, evaluated in the page.
//
// viser keeps its viewer state on window.__viserMutable (an e2e hook in its
// App.tsx): the three.js camera, the CameraControls instance, and
// nodeRefFromName, a map from scene-tree path to Object3D. Everything here
// projects world geometry through that camera, so a click lands on a thing's
// actual pixel and a drag starts on a gizmo's actual arrow wherever the camera
// is. viser's world is z-up and three's is y-up; the root node "" carries the
// rotation between them.
//
// The transform gizmos are drei PivotControls. Each arrow is a group holding a
// visible cone and an invisible hit cylinder (radius scale/20*1.4, a few pixels
// on screen), which is what a pointerdown has to land on; drei paints the
// hovered arrow's cone #ffff40, which is how a press is confirmed before it is
// made. Index order is traversal order, the same for the cylinders and the
// cones.
//
// Evaluated as an expression: returns true once viser has mounted its canvas.
() => {
  const m = window.__viserMutable;
  if (!m || !m.camera || !m.canvas) return false;
  const V = m.camera.position.constructor;
  const Q = m.camera.quaternion.constructor;
  const HOVER = 'ffff40';
  const hits = (name) => {
    const node = m.nodeRefFromName[name];
    const out = [];
    if (!node) return out;
    node.traverse((o) => {
      if (o.isMesh && o.geometry && o.geometry.type === 'CylinderGeometry' && !o.visible) out.push(o);
    });
    return out;
  };
  const cones = (name) => {
    const node = m.nodeRefFromName[name];
    const out = [];
    if (!node) return out;
    node.traverse((o) => {
      if (o.isMesh && o.geometry && o.geometry.type === 'ConeGeometry') out.push(o);
    });
    return out;
  };
  const visible = (o) => {
    for (let p = o; p; p = p.parent) if (!p.visible) return false;
    return true;
  };
  const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
  // Folder headers and labels are matched on their own text, never a parent's.
  const leaves = (root, text) =>
    Array.from(root.querySelectorAll('*')).filter(
      (e) => e.childElementCount === 0 && e.textContent.trim() === text,
    );
  const rect = (el) => {
    const r = el.getBoundingClientRect();
    return {x: r.x, y: r.y, w: r.width, h: r.height, cx: r.x + r.width / 2, cy: r.y + r.height / 2};
  };
  const api = {
    toPixel(v) {
      const p = v.clone().project(m.camera);
      const r = m.canvas.getBoundingClientRect();
      return [r.left + ((p.x + 1) / 2) * r.width, r.top + ((1 - p.y) / 2) * r.height, p.z];
    },
    toThree(xyz) {
      const root = m.nodeRefFromName[''];
      const v = new V(xyz[0], xyz[1], xyz[2]);
      if (root) v.applyMatrix4(root.matrixWorld);
      return v;
    },
    fromThree(v) {
      const root = m.nodeRefFromName[''];
      const w = v.clone();
      if (root) w.applyMatrix4(root.matrixWorld.clone().invert());
      return [w.x, w.y, w.z];
    },
    projectWorld(xyz) {
      return this.toPixel(this.toThree(xyz));
    },
    nodeWorld(name) {
      const node = m.nodeRefFromName[name];
      if (!node) return null;
      const v = new V();
      node.getWorldPosition(v);
      return this.fromThree(v);
    },
    nodeVisible(name) {
      const node = m.nodeRefFromName[name];
      return !!node && visible(node);
    },
    names() {
      return Object.keys(m.nodeRefFromName);
    },
    canvasRect() {
      const r = m.canvas.getBoundingClientRect();
      return [r.left, r.top, r.width, r.height];
    },
    renderer() {
      const gl = m.canvas.getContext('webgl2') || m.canvas.getContext('webgl');
      const e = gl && gl.getExtension('WEBGL_debug_renderer_info');
      return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : '';
    },
    camera() {
      const t = m.cameraControl ? m.cameraControl.getTarget(new V()) : new V();
      return {eye: this.fromThree(m.camera.position), at: this.fromThree(t), fov: m.camera.fov};
    },
    // Each arrow of a gizmo: the hit cylinder's centre, the gizmo's origin and
    // the cone's tip in pixels, the axis in viser world, and its length [m].
    gizmoAxes(name) {
      const node = m.nodeRefFromName[name];
      if (!node || !visible(node)) return null;
      const origin = new V();
      node.getWorldPosition(origin);
      const zero = this.fromThree(new V(0, 0, 0));
      return hits(name).map((o) => {
        const wp = new V();
        o.getWorldPosition(wp);
        const q = new Q();
        o.getWorldQuaternion(q);
        const dir = new V(0, 1, 0).applyQuaternion(q).normalize();
        const tip = new V();
        o.parent.traverse((s) => {
          if (s.isMesh && s.geometry.type === 'ConeGeometry') s.getWorldPosition(tip);
        });
        return {
          centre: this.toPixel(wp).slice(0, 2),
          origin: this.toPixel(origin).slice(0, 2),
          tip: this.toPixel(tip).slice(0, 2),
          world_dir: this.fromThree(dir).map((v, i) => v - zero[i]),
          length_m: o.geometry.parameters.height,
        };
      });
    },
    // Which arrow the pointer is over now, by its cone's hover colour; -1 for none.
    hoveredArrow(name) {
      return cones(name).findIndex((o) => o.material && o.material.color && o.material.color.getHexString() === HOVER);
    },
    // Where to put the pointer so that an arrow pressed at (px, py) has been
    // dragged `metres` along its own axis: the point on the axis nearest the
    // press ray, moved `metres`, projected back. Exact under perspective,
    // where a pixels-per-metre measured at the arrow overshoots a long drag.
    axisTarget(name, index, px, py, metres) {
      const o = hits(name)[index];
      if (!o) return null;
      const wp = new V();
      o.getWorldPosition(wp);
      const q = new Q();
      o.getWorldQuaternion(q);
      const dir = new V(0, 1, 0).applyQuaternion(q).normalize();
      const r = m.canvas.getBoundingClientRect();
      const ro = new V();
      m.camera.getWorldPosition(ro);
      const nd = new V(((px - r.left) / r.width) * 2 - 1, -(((py - r.top) / r.height) * 2 - 1), 0.5).unproject(m.camera);
      const rd = nd.sub(ro).normalize();
      const w0 = wp.clone().sub(ro);
      const b = dir.dot(rd);
      const d = dir.dot(w0);
      const e = rd.dot(w0);
      const t = (b * e - d) / Math.max(1e-9, 1 - b * b);
      return this.toPixel(wp.clone().add(dir.clone().multiplyScalar(t + metres))).slice(0, 2);
    },
    // Resolves once the stamp node shows frame `index`: the stamp is sent after
    // the frame's state, viser applies a batch in order and sets node poses in
    // the same useFrame, so a stamp in place is a state in place. Then one
    // render is asked for and two animation frames waited, so the canvas has
    // drawn it. Returns the milliseconds spent, or -1 on a timeout.
    async waitFrame(stamp, index, timeoutMs) {
      const t0 = performance.now();
      while (performance.now() - t0 < timeoutMs) {
        const node = m.nodeRefFromName[stamp];
        if (node && Math.abs(node.position.x - index) < 0.25) {
          m.requestRender();
          await raf();
          await raf();
          return performance.now() - t0;
        }
        m.requestRender();
        await raf();
      }
      return -1;
    },
    // ------------------------------------------------------------ the sidebar
    // The folder whose header reads `title` (or starts with it, for "Arm: right
    // (work)"). viser's header is a Paper holding the label as a bare text node
    // and a chevron icon -- up when the fold is open -- and its parent is the fold.
    folder(title) {
      const own = (e) =>
        Array.from(e.childNodes)
          .filter((n) => n.nodeType === 3)
          .map((n) => n.textContent)
          .join('')
          .trim();
      const head = Array.from(document.querySelectorAll('.mantine-Paper-root')).find((e) => own(e).startsWith(title));
      return head ? {box: head.parentElement, head} : null;
    },
    folderHeader(title) {
      const f = this.folder(title);
      if (!f) return null;
      f.head.scrollIntoView({block: 'nearest'});
      return {...rect(f.head), open: this.folderOpen(title)};
    },
    folderOpen(title) {
      const f = this.folder(title);
      return !!f && !!f.head.querySelector('.tabler-icon-chevron-up');
    },
    button(text, within) {
      const root = within ? (this.folder(within) || {}).box : document;
      if (!root) return null;
      const b = Array.from(root.querySelectorAll('button')).find((e) => e.textContent.trim() === text);
      return b ? rect(b) : null;
    },
    scrollIntoView(text, within) {
      const root = within ? (this.folder(within) || {}).box : document;
      if (!root) return false;
      const el =
        Array.from(root.querySelectorAll('button')).find((e) => e.textContent.trim() === text) || leaves(root, text)[0];
      if (!el) return false;
      el.scrollIntoView({block: 'center'});
      return true;
    },
    // The value of the read-only row labelled `label`.
    text(label, within) {
      const root = within ? (this.folder(within) || {}).box : document;
      if (!root) return '';
      const l = leaves(root, label).find((e) => e.tagName === 'LABEL' || e.tagName === 'P' || e.tagName === 'DIV');
      if (!l) return '';
      const row = l.closest('.mantine-Flex-root') || l.parentElement.parentElement;
      const inp = row && row.querySelector('input, textarea');
      return inp ? inp.value : '';
    },
    // A slider's track, thumb and aria values, by its label.
    slider(label, within) {
      const root = within ? (this.folder(within) || {}).box : document;
      if (!root) return null;
      const lab = Array.from(root.querySelectorAll('label')).find(
        (e) => e.textContent.trim() === label && e.getBoundingClientRect().width > 0,
      );
      if (!lab) return null;
      const box = (lab.htmlFor && document.getElementById(lab.htmlFor)) || lab.closest('.mantine-Flex-root');
      const track = box && box.querySelector('.mantine-Slider-track');
      if (!track) return null;
      const thumb = box.querySelector('.mantine-Slider-thumb');
      return {
        track: rect(track),
        thumb: thumb ? rect(thumb) : null,
        value: thumb ? parseFloat(thumb.getAttribute('aria-valuenow')) : null,
        min: thumb ? parseFloat(thumb.getAttribute('aria-valuemin')) : null,
        max: thumb ? parseFloat(thumb.getAttribute('aria-valuemax')) : null,
      };
    },
    // A dropdown's input by its label, and once open, an option by its text.
    select(label, within) {
      const root = within ? (this.folder(within) || {}).box : document;
      if (!root) return null;
      const lab = Array.from(root.querySelectorAll('label')).find((e) => e.textContent.trim() === label);
      if (!lab) return null;
      const input = (lab.htmlFor && document.getElementById(lab.htmlFor)) || lab.closest('.mantine-Flex-root').querySelector('input');
      return input ? {...rect(input), value: input.value} : null;
    },
    option(text) {
      const o = Array.from(document.querySelectorAll('[role=option]')).find(
        (e) => e.textContent.trim() === text && e.getBoundingClientRect().height > 0,
      );
      return o ? rect(o) : null;
    },
    checkbox(label, within) {
      const root = within ? (this.folder(within) || {}).box : document;
      if (!root) return null;
      const lab = Array.from(root.querySelectorAll('label')).find((e) => e.textContent.trim() === label);
      if (!lab) return null;
      const row = lab.closest('.mantine-Flex-root') || lab.parentElement;
      const input = row && row.querySelector('input[type=checkbox]');
      return input ? {...rect(input), checked: input.checked} : null;
    },
    panelRect() {
      const panel = document.querySelector('.mantine-ScrollArea-viewport') || document.querySelector('[class*=ControlPanel]');
      return panel ? rect(panel) : null;
    },
  };
  window.__rss = api;
  return true;
};
