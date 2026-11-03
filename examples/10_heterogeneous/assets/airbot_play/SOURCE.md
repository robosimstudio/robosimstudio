# AIRBOT Play model provenance

The MJCF and OBJ meshes in this directory come from [DISCOVERSE](https://github.com/discoverse-dev/DISCOVERSE), commit `d67f47c084aba0e0cf422a8725235f8b9238655a`, path `models/mjcf/manipulator/new_airbot_play/`. The accompanying `LICENSE` is the upstream MIT license.

The local `airbot_play.xml` drives each parallel finger with a direct position actuator instead of driving one finger through a general actuator and an equality constraint, and removes that equality constraint. Newton's MJCF importer exposes direct position actuators as joint targets, which RSS needs for scene control. The hand is otherwise the source model's: each finger carries upstream's two collision boxes -- the 10.5 x 4 x 40 mm pad and the finger body below it -- with the model's own collision friction. No pad adapter, shelf or added contact geometry is present.
