# SPDX-License-Identifier: Apache-2.0
"""The scene under the live page. Edit, save; serve.py rebuilds it in place.

The contract is one line: when this file has run, ``scene`` is an
``rss.Scene``. The server builds it (``scene.build(control_mode=
"joint_position")``, or the ``sim`` this file bound, if it bound one) and
swaps it into the open page. Poses are in the cell's work-surface frame.
"""

import robosimstudio as rss

scene = rss.Scene(cell="void", name="live")
scene.add(rss.robot("yam_i2rt"))  # a scene needs something that moves before it will build

if __name__ == "__main__":
    rss.view(scene.build(control_mode="joint_position"))
