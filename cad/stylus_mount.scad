// SO-101 follower wrist adapter with an open, flat stylus snap cradle.
// Horn interface measured from the pinned upstream wrist STEP.

/* [Stylus fit] */
stylus_diameter = 9; // mm, estimated from the photo
seat_clearance = -0.1; // light clip preload; adjust after a trial print
snap_gap = 8; // top opening narrower than the barrel

/* [Flat cradle] */
cradle_start = 18; // clear of the horn screws
cradle_length = 55;
cradle_width = 18;
base_thickness = 3;
stop_thickness = 3;
stop_height = 18; // solid wall at the arm end of the base
clip_width = 6;
clip_wall = 1.5;
rear_clip_offset = 18; // from cradle start, clear of the tapered pen nose
clip_spacing = 31;
bridge_width = 18; // clearance to the upstream wrist holder

/* [Wrist interface] */
horn_pcd = 14;
screw_clearance = 3.2;
center_clearance = 5.4;
plate_thickness = 3;
horn_lip_depth = 1;

/* [Hidden] */
$fn = 96;
hole_offset = horn_pcd / (2 * sqrt(2));
plate_top = horn_lip_depth + plate_thickness;
seat_diameter = stylus_diameter + seat_clearance;
clip_outer = seat_diameter + 2 * clip_wall;
base_top = horn_lip_depth + base_thickness;
pen_axis = base_top + seat_diameter/2 + 0.2;
assert(cradle_start > hole_offset + 4, "Keep the cradle clear of horn screws");
assert(stylus_diameter > 0 && seat_diameter > 0 && clip_wall > 0);
assert(snap_gap > 0 && snap_gap < stylus_diameter && snap_gap < seat_diameter);
assert(stop_height >= pen_axis + stylus_diameter/2,
       "The stop wall must cover the pen end");
assert(stop_thickness > 0 && base_thickness > 0 && clip_width > 0);
assert(cradle_width > clip_outer);
assert(rear_clip_offset - clip_width/2 >= stop_thickness + 10,
       "Keep the rear clip ahead of the tapered pen nose");
assert(clip_spacing > clip_width);
assert(rear_clip_offset + clip_spacing + clip_width/2 <= cradle_length);
assert(bridge_width > 2 * hole_offset + screw_clearance);
assert(plate_thickness > 0 && horn_lip_depth > 0);

module snap_clip(center_x) {
    translate([center_x - clip_width/2, 0, pen_axis])
        rotate([90,0,90])
            linear_extrude(height=clip_width)
                difference() {
                    circle(d=clip_outer);
                    circle(d=seat_diameter);
                    // The mouth widens above the retaining lips for insertion.
                    polygon([[-snap_gap/2,0], [snap_gap/2,0],
                             [snap_gap/2,seat_diameter/3],
                             [clip_outer/2,clip_outer/2+0.1],
                             [-clip_outer/2,clip_outer/2+0.1],
                             [-snap_gap/2,seat_diameter/3]]);
                }
}

module cradle() {
    translate([cradle_start,-cradle_width/2,horn_lip_depth])
        cube([cradle_length,cradle_width,base_thickness]);
    translate([cradle_start,-cradle_width/2,horn_lip_depth])
        cube([stop_thickness,cradle_width,stop_height-horn_lip_depth]);
    for (offset=[rear_clip_offset,rear_clip_offset+clip_spacing])
        snap_clip(cradle_start+offset);
}

difference() {
    union() {
        translate([0,0,horn_lip_depth]) cylinder(d=24,h=plate_thickness);
        difference() {
            cylinder(d=24,h=horn_lip_depth+0.01);
            translate([0,0,-0.1]) cylinder(d=20,h=horn_lip_depth+0.2);
        }
        // Low bridge keeps screw heads accessible from above.
        translate([0,-bridge_width/2,horn_lip_depth])
            cube([cradle_start+1,bridge_width,plate_thickness]);
        cradle();
    }
    for (x=[-hole_offset,hole_offset], y=[-hole_offset,hole_offset])
        translate([x,y,-0.1]) cylinder(d=screw_clearance,h=plate_top+0.2);
    translate([0,0,-0.1]) cylinder(d=center_clearance,h=plate_top+0.2);
    translate([0,0,-0.1]) cylinder(d=20,h=horn_lip_depth+0.1);
}
