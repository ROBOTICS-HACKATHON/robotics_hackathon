# Upstream reference CAD

Unmodified files from https://github.com/TheRobotStudio/SO-ARM100 at commit `a758567c3978dfeefe282ede0500085a48fe8f78`, retrieved 2026-10-14. The upstream Apache-2.0 license is preserved in `LICENSE.upstream`. These are inspection/reference parts, not the custom stylus mount.

| Local file | Upstream path |
| --- | --- |
| Moving_Jaw_SO101.stl | STL/SO101/Individual/Moving_Jaw_SO101.stl |
| Moving_Jaw_SO101.step | STEP/SO101/Follower_Specific/Moving_Jaw_SO101.step |
| Wrist_Roll_Follower_SO101.stl | STL/SO101/Individual/Wrist_Roll_Follower_SO101.stl |
| Wrist_Roll_Follower_SO101.step | STEP/SO101/Follower_Specific/Wrist_Roll_Follower_SO101.step |
| LICENSE.upstream | LICENSE |

Check the downloaded files from the repository root:

```sh
sha256sum -c cad/reference/SHA256SUMS
```

The custom adapter source is in `../stylus_mount.scad`; it uses dimensions measured from the upstream wrist STEP. No upstream file has been edited.
