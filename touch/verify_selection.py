"""Are exactly the right tiles selected?  (the check between "tiles tapped" and "press Verify")

The solver says which tiles to select; the arm taps them; then the camera looks again and this module reads
which tiles actually show the blue check disc. A tap toggles a tile, so a difference is fixed by tapping
ONLY the tiles that differ (never by re-tapping everything).

  chk = check_selection(img, expected={1, 3, 8})
  chk.ok          # selected set == expected set
  chk.to_tap      # tiles to tap to fix it (extra ones to untick + missing ones to tick)

`observe_stable` reads twice, a moment apart, and only trusts a reading both looks agree on, so a blurry
frame or a half-finished animation can never trigger a corrective tap.
"""

import time
from dataclasses import dataclass, field

from pagestate import GridNotFound, find_grid, read_selection, selected_set


@dataclass
class SelectionCheck:
    expected: set
    observed: set
    grid: object
    reads: list = field(default_factory=list)

    @property
    def ok(self):
        return self.observed == self.expected

    @property
    def extra(self):          # selected but should not be
        return self.observed - self.expected

    @property
    def missing(self):        # should be selected but is not
        return self.expected - self.observed

    @property
    def to_tap(self):
        return sorted(self.extra | self.missing)

    @property
    def doubtful(self):
        """Tiles where the disc and the white rim disagree (the reading is less certain there)."""
        return [r.index for r in self.reads if not r.agree]

    def __str__(self):
        if self.ok:
            return f"selection OK {sorted(self.observed)}"
        return (f"selection differs: expected {sorted(self.expected)}, seen {sorted(self.observed)} "
                f"(untick {sorted(self.extra)}, tick {sorted(self.missing)})")


def check_selection(img, expected, grid=None):
    """Read the selection in a straightened screenshot. Raises GridNotFound if the dialog is not clearly there."""
    grid = grid or find_grid(img)
    reads = read_selection(img, grid)
    return SelectionCheck(set(expected), selected_set(reads), grid, reads)


def observe_stable(look, expected, wait=0.6, looks=2, max_looks=5, sleep=time.sleep):
    """Look until `looks` consecutive readings agree. Returns the last SelectionCheck, or None if the screen
    never gave a steady, readable dialog (caller must then NOT tap anything)."""
    seen = []
    for _ in range(max_looks):
        img = look()
        if img is not None:
            try:
                seen.append(check_selection(img, expected))
            except GridNotFound:
                seen.clear()
        if len(seen) >= looks and all(s.observed == seen[-1].observed for s in seen[-looks:]):
            return seen[-1]
        sleep(wait)
    return None
