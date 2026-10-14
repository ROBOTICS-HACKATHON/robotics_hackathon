# SO-101 flat snap cradle

A short, open stylus holder replacing the follower's gripper housing on the **wrist-roll motor 5 horn**. The pen snaps down into two clips on a flat base, with most of its length exposed. A solid wall across the arm end of the base prevents the pen from sliding backward under screen-contact pressure. No tape or full-length sleeve is needed.

- [Printable STL](print/stylus_mount.stl), default size **85 × 24 × 18 mm**.
- [Editable OpenSCAD source](stylus_mount.scad), with Customizer parameters.
- [Preview](preview.png).
- [Original jaw and wrist CAD](reference/).

## Build and customize

From the repository root, with OpenSCAD installed:

```sh
openscad cad/stylus_mount.scad
mise run cad:build
```

The photo has no scale. A **9 mm barrel and roughly 145 mm total length** are estimates, excluding the black tape. Only the rear 55 mm holder is printed; the entire pen does not need to be enclosed. Remove the tape before fitting. Exact tip reach depends on the pen length with its ink point retracted and its rear end seated against the wall.

The two clips are 6 mm wide and 31 mm apart. Their default seat is Ø8.9 mm for light preload on the estimated Ø9 mm barrel, with an 8 mm mouth and 1.5 mm walls. The tapered entry lets the barrel snap down through the narrower opening. These are trial-fit dimensions, not validated snap-fit tolerances. Increase `seat_clearance` if tight, decrease it if loose, and adjust `stylus_diameter` to the actual barrel. Do not force insertion. `snap_gap`, `clip_wall`, clip spacing, `stop_thickness` and `stop_height` are customizable. The default wall is 3 mm thick, spans the 18 mm base width, and reaches 18 mm above the print bed.

The mouth retains the barrel radially; axial pull-away retention relies on clip friction. The rear stop only blocks backward travel under screen pressure. Check both directions before using the robot. Keep the wrist-interface values and 18 mm bridge unchanged unless hardware measurements justify a change.

## Print and assemble

1. Use PETG as a starting material for the flexible clips. Start with 0.2 mm layers, 4 walls and a brim. Orient the pen axis vertically so each C-shaped clip arc lies in the layer plane; use supports under both clips, the projecting base, rear stop and wrist attachment, keeping them out of clip seats and mouths. Flat-base printing is simpler, but opening the clips then loads layer bonds. Neither orientation has been physically validated.
2. With power off, remove the complete gripper housing from motor 5's horn. Retain the horn and center screw. Reuse the four M3×6 horn screws if they engage correctly through the 3 mm plate; check thread depth before using longer screws.
3. Keep gripper servo **motor 6 electrically connected and securely stowed**, with cable strain relieved and shaft clear. The existing six-motor LeRobot configuration expects it. Keep commands to that servo stationary.
4. Check seating, screw alignment and access, and wrist clearance before tightening. Retract the ink point. Seat the retracted ink end against the solid rear wall, then snap the bare barrel into both clips. The wall has no hole or notch, so do not fit the pen with the ink point extended. The pocket clip must clear the printed clips. Confirm the rubber touchscreen end points away from the arm.
5. Check repeated insertion, radial lift-out and axial pull-away by hand, including light drag in both directions. Adjust fit if the pen shifts. Test touchscreen response with nobody touching the barrel; passive styluses may depend on hand coupling.
6. For a laptop, brace the screen and base. Approach close to perpendicular to the glass, with only the rubber tip contacting it. Check keyboard/bezel clearance at the lower screen and corners. Retract away from the glass before moving sideways. Recalibrate for the new tool and start with light contact; the mount has no spring compliance or force limiting.

**Physical fit, clip retention, printer tolerances, touchscreen response and robot motion remain unverified.** Digital checks validate the mesh, open seats, solid stop wall and unchanged screw holes. Prior upstream checks established the horn interface and 18 mm bridge clearance at the reference pose; this revised cradle has not been checked through a complete wrist sweep.

## Source and mounting dimensions

The [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101) links the [TheRobotStudio/SO-ARM100 hardware repository](https://github.com/TheRobotStudio/SO-ARM100). References are pinned to commit [`a758567c3978dfeefe282ede0500085a48fe8f78`](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78), retrieved 2026-10-14.

The interface was measured from the cylindrical faces of `STEP/SO101/Follower_Specific/Wrist_Roll_Follower_SO101.step`, rather than estimated from the photograph:

| Feature | Measurement / adapter value |
| --- | --- |
| Four M3 clearance holes | Ø3.2 mm |
| Bolt circle | Ø14 mm, four holes at 45°, 135°, 225°, 315° |
| Square hole spacing | 9.899494937 mm in X and Y |
| Source hole centers | X = ±4.949747468 mm; Y = −0.218214479 ±4.949747468 mm |
| Source plate at mounting holes | 3 mm thick |
| Central screw clearance | Ø5.4 mm |
| Source horn-seat recess | Ø20 mm, 1 mm deep |
| Adapter locating lip | Ø24 outside, Ø20 inside, 1 mm deep |
| Bridge width | 18 mm, reduced to clear the source wrist holder |

The source coordinates were recentered on the horn axis for this model. The original housing has additional cutouts and rounded surfaces; this adapter is a newly modeled replacement, not the original housing mesh. Assembly instructions specify four M3×6 screws attaching the housing to the wrist motor horn.

Unmodified reference files and their Apache-2.0 license are in `reference/`; exact upstream paths and SHA-256 checksums are recorded there.
