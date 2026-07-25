import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


def _unique_colors(arr):
    return set(np.unique(arr)) - {0}


def _detect_color_inversion(training_pairs):
    """Color Inversion"""
    all_input_colors = set()
    all_output_colors = set()
    for inp, out in training_pairs:
        if inp.shape != out.shape:
            return None
        inp_colors = _unique_colors(inp)
        out_colors = _unique_colors(out)
        if len(inp_colors) != 2 or len(out_colors) != 1:
            return None
        all_input_colors |= inp_colors
        all_output_colors |= out_colors
    candidates = all_input_colors - all_output_colors
    if len(candidates) == 1:
        return candidates.pop()
    return None

def _solve_color_inversion(training_pairs, test_input):
    marker = _detect_color_inversion(training_pairs)
    others = _unique_colors(test_input) - {marker}
    if not others:
        return None
    other = others.pop()
    result = np.zeros_like(test_input)
    result[test_input == marker] = other
    return result


def _detect_color_swap(training_pairs):
    """Color Swap"""
    mapping = {}
    for inp, out in training_pairs:
        if inp.shape != out.shape:
            return None
        changed = inp != out
        for from_c in np.unique(inp[changed]):
            from_c = int(from_c)
            to_vals = set(np.unique(out[inp == from_c]).tolist())
            if len(to_vals) != 1:
                return None
            to_c = to_vals.pop()
            if from_c in mapping and mapping[from_c] != to_c:
                return None
            mapping[from_c] = to_c
    if len(mapping) == 1:
        return list(mapping.items())[0]
    return None

def _solve_color_swap(training_pairs, test_input):
    from_c, to_c = _detect_color_swap(training_pairs)
    output = test_input.copy()
    output[test_input == from_c] = to_c
    return output



# Unified from AND/OR/NOR: the grid is split by a full row or column of a
# single non-zero value into two equal halves, then a per-cell boolean op
# combines them. Generalizes the three original detectors and adds XOR, plus
# a color-preserving output mode (keep each half's own color instead of a
# single flat fill).

def _find_separator(grid, orient):
    """Index of a full row ('row') or column ('col') of one non-zero value
    that sits at the exact center, so the two halves are equal-sized. The
    center requirement avoids picking a stray uniform data line (e.g. a row
    that happens to be a solid color) instead of the real separator."""
    n = grid.shape[0] if orient == "row" else grid.shape[1]
    for i in range(n):
        if i != n - 1 - i:
            continue
        line = grid[i, :] if orient == "row" else grid[:, i]
        vals = set(line.tolist())
        if len(vals) == 1 and 0 not in vals:
            return i
    return None

def _split(grid, sep, orient):
    if orient == "row":
        return grid[:sep, :], grid[sep + 1:, :]
    return grid[:, :sep], grid[:, sep + 1:]

def _split_for_logic(grid, orient):
    """Two equal halves for a logic op. 'row'/'col' split on a center
    separator line; 'row_half'/'col_half' cut at the exact middle with no
    separator, accepted only when each half holds one distinct color (so an
    arbitrary same-shape grid can't be split into meaningless halves)."""
    if orient in ("row", "col"):
        sep = _find_separator(grid, orient)
        if sep is None:
            return None
        a, b = _split(grid, sep, orient)
    else:
        H, W = grid.shape
        if orient == "row_half":
            if H % 2:
                return None
            a, b = grid[:H // 2, :], grid[H // 2:, :]
        else:
            if W % 2:
                return None
            a, b = grid[:, :W // 2], grid[:, W // 2:]
        ca, cb = _unique_colors(a), _unique_colors(b)
        if len(ca) != 1 or len(cb) != 1 or ca == cb:
            return None
    if a.shape != b.shape:
        return None
    return a, b

def _logic_mask(a, b, op):
    am, bm = a != 0, b != 0
    if op == "and":
        return am & bm
    if op == "or":
        return am | bm
    if op == "xor":
        return am ^ bm
    return ~(am | bm)  # nor

def _logic_build(a, b, op, mode, flat_color):
    mask = _logic_mask(a, b, op)
    result = np.zeros_like(a)
    if mode == "flat":
        result[mask] = flat_color
    else:  # preserve: a's color wins, else b's color
        from_a = mask & (a != 0)
        result[from_a] = a[from_a]
        from_b = mask & (a == 0) & (b != 0)
        result[from_b] = b[from_b]
    return result

def _solve_binary_logic(training_pairs, test_input):
    """Try each orientation/op/coloring; return the test output for the
    first combination that reproduces every training output exactly."""
    for orient in ("row", "col", "row_half", "col_half"):
        splits = []
        ok = True
        for inp, out in training_pairs:
            halves = _split_for_logic(inp, orient)
            if halves is None or halves[0].shape != out.shape:
                ok = False
                break
            splits.append((halves[0], halves[1], out))
        if not ok:
            continue

        out_colors = set()
        for _a, _b, out in splits:
            out_colors |= _unique_colors(out)
        flat_color = out_colors.pop() if len(out_colors) == 1 else None

        for op in ("and", "or", "xor", "nor"):
            for mode in ("flat", "preserve"):
                if mode == "flat" and flat_color is None:
                    continue
                if mode == "preserve" and op == "nor":
                    continue
                if all(np.array_equal(_logic_build(a, b, op, mode, flat_color), out)
                       for a, b, out in splits):
                    halves = _split_for_logic(test_input, orient)
                    if halves is None:
                        return None
                    return _logic_build(halves[0], halves[1], op, mode, flat_color)
    return None

# Generalized from rotation: any single geometric transform whose result
# reproduces every training output. Rotations are tried first so a pure
# rotation problem still resolves to a rotation; transpose/flips extend the
# family (e.g. Set C transpose problems).
_GEO_TRANSFORMS = [
    ("rot90", lambda a: np.rot90(a, 1)),
    ("rot180", lambda a: np.rot90(a, 2)),
    ("rot270", lambda a: np.rot90(a, 3)),
    ("fliplr", np.fliplr),
    ("flipud", np.flipud),
    ("transpose", np.transpose),
    ("anti_transpose", lambda a: np.fliplr(np.flipud(np.transpose(a)))),
]

def _detect_geometric(training_pairs):
    """Returns the transform fn if one reproduces output for all pairs."""
    for _name, fn in _GEO_TRANSFORMS:
        if all(np.array_equal(fn(inp), out) for inp, out in training_pairs):
            return fn
    return None

def _solve_geometric(training_pairs, test_input):
    fn = _detect_geometric(training_pairs)
    return fn(test_input)

# Connected components of equal-colored non-zero cells (4- or 8-adjacency).
def _components(grid, diag=False):
    H, W = grid.shape
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        neighbors += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0 and not seen[i, j]:
                color = grid[i, j]
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and not seen[nr, nc] and grid[nr, nc] == color):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append((color, cells))
    return comps


# --- Hollow rectangles: keep each solid block's border, clear its interior ---

def _apply_hollow(grid):
    result = grid.copy()
    for _color, cells in _components(grid):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        minr, maxr, minc, maxc = min(rs), max(rs), min(cs), max(cs)
        for r, c in cells:
            if minr < r < maxr and minc < c < maxc:
                result[r, c] = 0
    return result

def _detect_hollow(pairs):
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        if not np.array_equal(_apply_hollow(inp), out):
            return False
    return True

def _solve_hollow(pairs, test_input):
    return _apply_hollow(test_input)


# --- Diagonal X: a single pixel projects both diagonals to the edges ---

def _apply_diagonal_x(grid):
    nz = np.argwhere(grid != 0)
    if len(nz) != 1:
        return None
    r0, c0 = nz[0]
    color = grid[r0, c0]
    result = np.zeros_like(grid)
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            if abs(i - r0) == abs(j - c0):
                result[i, j] = color
    return result

def _detect_diagonal_x(pairs):
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_diagonal_x(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_diagonal_x(pairs, test_input):
    return _apply_diagonal_x(test_input)


# --- Block fill: each marker fills the aligned block that contains it ---

def _apply_blockfill(grid, bh, bw, fill):
    result = np.zeros_like(grid)
    for r, c in np.argwhere(grid != 0):
        br, bc = (r // bh) * bh, (c // bw) * bw
        result[br:br + bh, bc:bc + bw] = fill
    return result

def _blockfill_params(pairs):
    marker = fill = None
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        ic, oc = _unique_colors(inp), _unique_colors(out)
        if len(ic) != 1 or len(oc) != 1:
            return None
        m, f = ic.pop(), oc.pop()
        if marker is None:
            marker, fill = m, f
        elif (marker, fill) != (m, f):
            return None

    from math import gcd
    g_h = pairs[0][0].shape[0]
    g_w = pairs[0][0].shape[1]
    for inp, _out in pairs[1:]:
        g_h = gcd(g_h, inp.shape[0])
        g_w = gcd(g_w, inp.shape[1])
    divisors = lambda n: [d for d in range(1, n + 1) if n % d == 0]
    for bh in divisors(g_h):
        for bw in divisors(g_w):
            if bh == 1 and bw == 1:
                continue
            if all(np.array_equal(_apply_blockfill(inp, bh, bw, fill), out)
                   for inp, out in pairs):
                return (bh, bw, fill)
    return None

def _detect_blockfill(pairs):
    return _blockfill_params(pairs) is not None

def _solve_blockfill(pairs, test_input):
    bh, bw, fill = _blockfill_params(pairs)
    return _apply_blockfill(test_input, bh, bw, fill)


# --- Quad mirror: tile input with its mirror images into a 2H x 2W grid ---

def _apply_quad_mirror(g):
    top = np.hstack([g, np.fliplr(g)])
    return np.vstack([top, np.flipud(top)])

def _detect_quad_mirror(pairs):
    for inp, out in pairs:
        if out.shape != (inp.shape[0] * 2, inp.shape[1] * 2):
            return False
        if not np.array_equal(_apply_quad_mirror(inp), out):
            return False
    return True

def _solve_quad_mirror(pairs, test_input):
    return _apply_quad_mirror(test_input)


# --- Edge-match fill: fill a line when its two end cells share a color ---

def _apply_edge_fill(g, orient):
    out = g.copy()
    H, W = g.shape
    if orient == "row":
        for r in range(H):
            a, b = g[r, 0], g[r, W - 1]
            if a != 0 and a == b:
                out[r, :] = a
    else:
        for c in range(W):
            a, b = g[0, c], g[H - 1, c]
            if a != 0 and a == b:
                out[:, c] = a
    return out

def _detect_edge_fill(pairs):
    for orient in ("row", "col"):
        ok, changed = True, False
        for inp, out in pairs:
            if inp.shape != out.shape:
                ok = False
                break
            res = _apply_edge_fill(inp, orient)
            if not np.array_equal(res, out):
                ok = False
                break
            if not np.array_equal(res, inp):
                changed = True
        if ok and changed:
            return orient
    return None

def _solve_edge_fill(pairs, test_input):
    orient = _detect_edge_fill(pairs)
    return _apply_edge_fill(test_input, orient)


# --- Growing staircase: a 1-row bar expands by one cell per added row ---

def _apply_staircase(g):
    if g.shape[0] != 1:
        return None
    row = g[0]
    nz = np.nonzero(row)[0]
    if len(nz) == 0:
        return None
    k = len(nz)
    if not np.array_equal(nz, np.arange(k)):  # must be a left-aligned solid bar
        return None
    color = row[nz[0]]
    W = g.shape[1]
    nrows = W // 2
    if nrows < 1:
        return None
    out = np.zeros((nrows, W), dtype=g.dtype)
    for i in range(nrows):
        out[i, :min(k + i, W)] = color
    return out

def _detect_staircase(pairs):
    for inp, out in pairs:
        res = _apply_staircase(inp)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_staircase(pairs, test_input):
    return _apply_staircase(test_input)


# Cells reachable from the grid border through `passable` cells (4-adjacency).
def _flood_from_border(grid, passable):
    H, W = grid.shape
    reached = np.zeros((H, W), dtype=bool)
    stack = []
    for i in range(H):
        for j in (0, W - 1):
            if passable[i, j] and not reached[i, j]:
                reached[i, j] = True
                stack.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if passable[i, j] and not reached[i, j]:
                reached[i, j] = True
                stack.append((i, j))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W
                    and passable[nr, nc] and not reached[nr, nc]):
                reached[nr, nc] = True
                stack.append((nr, nc))
    return reached


# --- Corner crop: 4 corner markers bound a region; crop it, recolor shape ---

def _cornercrop_params(grid):
    for color in _unique_colors(grid):
        cells = np.argwhere(grid == color)
        if len(cells) != 4:
            continue
        rs = sorted(set(cells[:, 0].tolist()))
        cs = sorted(set(cells[:, 1].tolist()))
        if len(rs) == 2 and len(cs) == 2:
            corners = {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}
            if set(map(tuple, cells.tolist())) == corners:
                return color, rs[0], rs[1], cs[0], cs[1]
    return None

def _apply_cornercrop(grid):
    p = _cornercrop_params(grid)
    if p is None:
        return None
    marker, r0, r1, c0, c1 = p
    inner = _unique_colors(grid) - {marker}
    if len(inner) != 1:
        return None
    ic = inner.pop()
    region = grid[r0 + 1:r1, c0 + 1:c1]
    res = np.zeros_like(region)
    res[region == ic] = marker
    return res

def _detect_cornercrop(pairs):
    for inp, out in pairs:
        res = _apply_cornercrop(inp)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_cornercrop(pairs, test_input):
    return _apply_cornercrop(test_input)


# --- Move marker: one marker steps one cell toward the other ---

def _movemarker_mover(pairs):
    mover = None
    for inp, out in pairs:
        ic = _unique_colors(inp)
        if len(ic) != 2:
            return None
        moved = []
        for c in ic:
            pin, pout = np.argwhere(inp == c), np.argwhere(out == c)
            if len(pin) != 1 or len(pout) != 1:
                return None
            if not np.array_equal(pin[0], pout[0]):
                moved.append(c)
        if len(moved) != 1:
            return None
        if mover is None:
            mover = moved[0]
        elif mover != moved[0]:
            return None
    return mover

def _apply_movemarker(grid, mover):
    ic = _unique_colors(grid)
    if mover not in ic or len(ic) != 2:
        return None
    anchor = (ic - {mover}).pop()
    pm = np.argwhere(grid == mover)[0]
    pa = np.argwhere(grid == anchor)[0]
    res = np.zeros_like(grid)
    res[pa[0], pa[1]] = anchor
    nr = pm[0] + int(np.sign(pa[0] - pm[0]))
    nc = pm[1] + int(np.sign(pa[1] - pm[1]))
    res[nr, nc] = mover
    return res

def _detect_movemarker(pairs):
    mover = _movemarker_mover(pairs)
    if mover is None:
        return False
    for inp, out in pairs:
        res = _apply_movemarker(inp, mover)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_movemarker(pairs, test_input):
    return _apply_movemarker(test_input, _movemarker_mover(pairs))


# --- Column sort: each color becomes a column, height = its count, sorted ---

def _apply_colsort(grid):
    ic = _unique_colors(grid)
    if not ic:
        return None
    counts = {c: int((grid == c).sum()) for c in ic}
    order = sorted(counts, key=lambda c: (-counts[c], c))
    nrows, ncols = max(counts.values()), len(order)
    res = np.zeros((nrows, ncols), dtype=grid.dtype)
    for j, c in enumerate(order):
        res[:counts[c], j] = c
    return res

def _detect_colsort(pairs):
    for inp, out in pairs:
        res = _apply_colsort(inp)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_colsort(pairs, test_input):
    return _apply_colsort(test_input)


# --- Panel overlay: split by uniform separators, overlay first-panel-wins ---

def _split_panels(grid):
    """Split into equal panels along whichever single separator color produces
    the most equal-sized panels. Keying on one color avoids over-splitting on a
    panel-internal column that happens to be solid."""
    H, W = grid.shape
    best = None
    for axis in ("col", "row"):
        n = W if axis == "col" else H
        line = lambda i: grid[:, i] if axis == "col" else grid[i, :]
        candidates = set()
        for i in range(n):
            vals = set(line(i).tolist())
            if len(vals) == 1 and 0 not in vals:
                candidates.add(line(i)[0])
        for s in candidates:
            seps = [i for i in range(n) if set(line(i).tolist()) == {s}]
            segs, start = [], 0
            for idx in seps + [n]:
                if idx > start:
                    seg = grid[:, start:idx] if axis == "col" else grid[start:idx, :]
                    segs.append(seg)
                start = idx + 1
            if len(segs) >= 2 and all(sg.shape == segs[0].shape for sg in segs):
                if best is None or len(segs) > len(best):
                    best = segs
    return best

def _apply_panel_overlay(grid):
    segs = _split_panels(grid)
    if not segs or len(segs) < 2:
        return None
    shp = segs[0].shape
    if any(s.shape != shp for s in segs):
        return None
    res = np.zeros(shp, dtype=grid.dtype)
    for s in segs:
        mask = (res == 0) & (s != 0)
        res[mask] = s[mask]
    return res

def _detect_panel_overlay(pairs):
    for inp, out in pairs:
        res = _apply_panel_overlay(inp)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_panel_overlay(pairs, test_input):
    return _apply_panel_overlay(test_input)


# --- Rooms: flood from border; outside background and enclosed get 2 colors ---

def _apply_rooms(grid, outside, enclosed):
    reached = _flood_from_border(grid, grid == 0)
    res = grid.copy()
    res[(grid == 0) & reached] = outside
    res[(grid == 0) & ~reached] = enclosed
    return res

def _rooms_params(pairs):
    # Wall color may differ per pair; only the outside/enclosed fill colors
    # need to be consistent, so they are all this solver tracks.
    params = None
    for inp, out in pairs:
        if len(_unique_colors(inp)) != 1:
            return None
        new = _unique_colors(out) - _unique_colors(inp)
        if len(new) != 2:
            return None
        H, W = inp.shape
        oc = None
        for i in range(H):
            for j in (0, W - 1):
                if inp[i, j] == 0:
                    oc = out[i, j]
                    break
            if oc is not None:
                break
        if oc is None or oc not in new:
            return None
        ec = (new - {oc}).pop()
        if params is None:
            params = (oc, ec)
        elif params != (oc, ec):
            return None
    for inp, out in pairs:
        if not np.array_equal(_apply_rooms(inp, *params), out):
            return None
    return params

def _detect_rooms(pairs):
    return _rooms_params(pairs) is not None

def _solve_rooms(pairs, test_input):
    return _apply_rooms(test_input, *_rooms_params(pairs))


# --- Closed loops: shape components that enclose background get recolored ---

def _apply_closedloop(grid, shape, new):
    reached = _flood_from_border(grid, grid != shape)
    enclosed = (grid != shape) & ~reached
    res = grid.copy()
    H, W = grid.shape
    for color, cells in _components(grid):
        if color != shape:
            continue
        closed = False
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and enclosed[nr, nc]:
                    closed = True
                    break
            if closed:
                break
        if closed:
            for r, c in cells:
                res[r, c] = new
    return res

def _closedloop_params(pairs):
    params = None
    for inp, out in pairs:
        ic = _unique_colors(inp)
        if len(ic) != 2:
            return None
        counts = {c: int((inp == c).sum()) for c in ic}
        bg = max(counts, key=counts.get)
        shape = (ic - {bg}).pop()
        new = _unique_colors(out) - _unique_colors(inp)
        if len(new) != 1:
            return None
        n = new.pop()
        if params is None:
            params = (shape, n)
        elif params != (shape, n):
            return None
    for inp, out in pairs:
        if not np.array_equal(_apply_closedloop(inp, *params), out):
            return None
    return params

def _detect_closedloop(pairs):
    return _closedloop_params(pairs) is not None

def _solve_closedloop(pairs, test_input):
    return _apply_closedloop(test_input, *_closedloop_params(pairs))


# --- Diagonal rays: each square emits a 45 deg ray in a learned direction ---

def _diag_ray_map(pairs):
    inp0, out0 = pairs[0]
    mapping = {}
    for color in _unique_colors(inp0):
        cells = np.argwhere(inp0 == color)
        r0, r1 = cells[:, 0].min(), cells[:, 0].max()
        c0, c1 = cells[:, 1].min(), cells[:, 1].max()
        ray = [(r, c) for r, c in np.argwhere(out0 == color).tolist()
               if not (r0 <= r <= r1 and c0 <= c <= c1)]
        if not ray:
            return None
        cr, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
        dr = -1 if ray[0][0] < cr else 1
        dc = -1 if ray[0][1] < cc else 1
        for r, c in ray:
            if not (np.sign(r - cr) == dr and np.sign(c - cc) == dc
                    and abs(r - cr) == abs(c - cc)):
                return None
        mapping[color] = (dr, dc)
    return mapping

def _apply_diag_rays(grid, mapping):
    res = grid.copy()
    H, W = grid.shape
    for color, (dr, dc) in mapping.items():
        cells = np.argwhere(grid == color)
        if len(cells) == 0:
            continue
        r0, r1 = cells[:, 0].min(), cells[:, 0].max()
        c0, c1 = cells[:, 1].min(), cells[:, 1].max()
        r = (r0 if dr < 0 else r1) + dr
        c = (c0 if dc < 0 else c1) + dc
        while 0 <= r < H and 0 <= c < W:
            res[r, c] = color
            r += dr
            c += dc
    return res

def _detect_diag_rays(pairs):
    m = _diag_ray_map(pairs)
    if not m:
        return False
    for inp, out in pairs:
        if not np.array_equal(_apply_diag_rays(inp, m), out):
            return False
    return True

def _solve_diag_rays(pairs, test_input):
    return _apply_diag_rays(test_input, _diag_ray_map(pairs))


# --- Arrow ray: a marker shoots a line in the arrow's pointing direction ---

def _arrow_params(grid):
    cols = _unique_colors(grid)
    if len(cols) != 2:
        return None
    counts = {c: int((grid == c).sum()) for c in cols}
    markers = [c for c in cols if counts[c] == 1]
    if len(markers) != 1:
        return None
    D = markers[0]
    C = (cols - {D}).pop()
    mr, mc = np.argwhere(grid == D)[0]
    acells = np.argwhere(grid == C)
    dr, dc = acells[:, 0].mean() - mr, acells[:, 1].mean() - mc
    if abs(dr) >= abs(dc):
        direction = (1 if dr > 0 else -1, 0)
    else:
        direction = (0, 1 if dc > 0 else -1)
    return D, direction

def _apply_arrow(grid):
    p = _arrow_params(grid)
    if p is None:
        return None
    D, (dr, dc) = p
    res = grid.copy()
    H, W = grid.shape
    mr, mc = np.argwhere(grid == D)[0]
    r, c = mr + dr, mc + dc
    while 0 <= r < H and 0 <= c < W:
        if res[r, c] == 0:
            res[r, c] = D
        r += dr
        c += dc
    return res

def _detect_arrow(pairs):
    for inp, out in pairs:
        res = _apply_arrow(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_arrow(pairs, test_input):
    return _apply_arrow(test_input)


# --- Box reflect: shapes inside a box mirror across their nearest wall ---

def _find_box(grid):
    # The box is the color whose bounding box strictly encloses the other color;
    # the frame need not be a complete perimeter (corners/brackets are enough).
    cols = _unique_colors(grid)
    if len(cols) != 2:
        return None
    for B in cols:
        other = (cols - {B}).pop()
        bc = np.argwhere(grid == B)
        oc = np.argwhere(grid == other)
        if len(oc) == 0:
            continue
        r0, r1 = bc[:, 0].min(), bc[:, 0].max()
        c0, c1 = bc[:, 1].min(), bc[:, 1].max()
        if (oc[:, 0].min() > r0 and oc[:, 0].max() < r1
                and oc[:, 1].min() > c0 and oc[:, 1].max() < c1):
            return B, r0, r1, c0, c1
    return None

def _apply_box_reflect(grid):
    box = _find_box(grid)
    if box is None:
        return None
    B, r0, r1, c0, c1 = box
    inner = _unique_colors(grid) - {B}
    if len(inner) != 1:
        return None
    S = inner.pop()
    res = grid.copy()
    res[grid == S] = 0
    H, W = grid.shape
    comps = [cells for color, cells in _components(grid, diag=True) if color == S]
    cents = [(sum(r for r, c in cl) / len(cl), sum(c for r, c in cl) / len(cl))
             for cl in comps]
    if not cents:
        return None
    rows = [cy for cy, cx in cents]
    colsx = [cx for cy, cx in cents]
    # the paired shapes are separated along one axis; reflect across the wall on
    # whichever side of the box centre each shape lies (along that axis).
    vertical = (max(rows) - min(rows)) >= (max(colsx) - min(colsx))
    rc, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    for cl, (cy, cx) in zip(comps, cents):
        for r, c in cl:
            if vertical:
                nr, nc = (2 * r0 - r, c) if cy < rc else (2 * r1 - r, c)
            else:
                nr, nc = (r, 2 * c0 - c) if cx < cc else (r, 2 * c1 - c)
            if 0 <= nr < H and 0 <= nc < W:
                res[nr, nc] = S
    return res

def _detect_box_reflect(pairs):
    for inp, out in pairs:
        res = _apply_box_reflect(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_box_reflect(pairs, test_input):
    return _apply_box_reflect(test_input)


# --- Box stamp: 4 corner markers become framed boxes joined by dotted lines ---

def _apply_boxstamp(grid, dot):
    cells = np.argwhere(grid != 0)
    if len(cells) != 4:
        return None
    cols = _unique_colors(grid)
    if len(cols) != 2:
        return None
    rs = sorted(set(cells[:, 0].tolist()))
    cs = sorted(set(cells[:, 1].tolist()))
    if len(rs) != 2 or len(cs) != 2:
        return None
    r0, r1 = rs
    c0, c1 = cs
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if set(map(tuple, cells.tolist())) != corners:
        return None
    res = np.zeros_like(grid)
    for mr, mc in corners:
        M = grid[mr, mc]
        O = (cols - {M}).pop()
        res[mr - 1:mr + 2, mc - 1:mc + 2] = O
        res[mr, mc] = M
    # dotted connectors along each edge: dot where distance to nearer box is even
    for R, ca, cb in [(r0, c0, c1), (r1, c0, c1)]:
        for q in range(ca + 2, cb - 1):
            if min(q - ca, cb - q) % 2 == 0:
                res[R, q] = dot
    for C, ra, rb in [(c0, r0, r1), (c1, r0, r1)]:
        for q in range(ra + 2, rb - 1):
            if min(q - ra, rb - q) % 2 == 0:
                res[q, C] = dot
    return res

def _boxstamp_dot(pairs):
    new = _unique_colors(pairs[0][1]) - _unique_colors(pairs[0][0])
    return new.pop() if len(new) == 1 else None

def _detect_boxstamp(pairs):
    dot = _boxstamp_dot(pairs)
    if dot is None:
        return False
    for inp, out in pairs:
        res = _apply_boxstamp(inp, dot)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_boxstamp(pairs, test_input):
    return _apply_boxstamp(test_input, _boxstamp_dot(pairs))


# --- Mirror tile 3x3: quad-mirror's big sibling; centre tile is the input,
# --- surrounding tiles are its mirror images (flipped toward the centre) ---

def _apply_mirror_tile3(g):
    rows = []
    for i in range(3):
        band = []
        for j in range(3):
            t = g
            if i != 1:
                t = np.flipud(t)
            if j != 1:
                t = np.fliplr(t)
            band.append(t)
        rows.append(np.hstack(band))
    return np.vstack(rows)

def _detect_mirror_tile3(pairs):
    for inp, out in pairs:
        if out.shape != (inp.shape[0] * 3, inp.shape[1] * 3):
            return False
        if not np.array_equal(_apply_mirror_tile3(inp), out):
            return False
    return True

def _solve_mirror_tile3(pairs, test_input):
    return _apply_mirror_tile3(test_input)


# --- Lattice histogram: blocks sit in a lattice of separator lines; one row
# --- per block color, its count wide, rows sorted by count ascending ---

def _lattice_color(grid):
    """The color that draws the lattice: forms at least one full uniform row
    and one full uniform column."""
    H, W = grid.shape
    row_colors = {int(grid[i, 0]) for i in range(H)
                  if grid[i, 0] != 0 and len(set(grid[i, :].tolist())) == 1}
    col_colors = {int(grid[0, j]) for j in range(W)
                  if grid[0, j] != 0 and len(set(grid[:, j].tolist())) == 1}
    both = row_colors & col_colors
    return both.pop() if len(both) == 1 else None

def _apply_lattice_hist(grid):
    lat = _lattice_color(grid)
    if lat is None:
        return None
    counts = {}
    for color, _cells in _components(grid):
        if color != lat:
            counts[color] = counts.get(color, 0) + 1
    if not counts:
        return None
    order = sorted(counts, key=lambda c: (counts[c], c))
    res = np.zeros((len(order), max(counts.values())), dtype=grid.dtype)
    for i, c in enumerate(order):
        res[i, :counts[c]] = c
    return res

def _detect_lattice_hist(pairs):
    for inp, out in pairs:
        res = _apply_lattice_hist(inp)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_lattice_hist(pairs, test_input):
    return _apply_lattice_hist(test_input)


# --- Count fill: count the noise dots inside a rectangular frame, then fill
# --- that many cells of a small fixed-size grid in reading order ---

def _rect_frame(grid):
    """Color whose cells form exactly the perimeter of a rectangle (>=3x3)."""
    for color in sorted(_unique_colors(grid)):
        cells = np.argwhere(grid == color)
        r0, r1 = cells[:, 0].min(), cells[:, 0].max()
        c0, c1 = cells[:, 1].min(), cells[:, 1].max()
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perim = ({(r, c) for r in range(r0, r1 + 1) for c in (c0, c1)}
                 | {(r, c) for c in range(c0, c1 + 1) for r in (r0, r1)})
        if set(map(tuple, cells.tolist())) == perim:
            return color, r0, r1, c0, c1
    return None

def _apply_count_fill(grid, out_shape):
    p = _rect_frame(grid)
    if p is None:
        return None
    frame, r0, r1, c0, c1 = p
    noise = _unique_colors(grid) - {frame}
    if len(noise) != 1:
        return None
    nc = noise.pop()
    inside = grid[r0 + 1:r1, c0 + 1:c1]
    n = int((inside == nc).sum())
    if n > out_shape[0] * out_shape[1]:
        return None
    res = np.zeros(out_shape, dtype=grid.dtype)
    res.reshape(-1)[:n] = nc
    return res

def _count_fill_shape(pairs):
    shapes = {out.shape for _inp, out in pairs}
    return shapes.pop() if len(shapes) == 1 else None

def _detect_count_fill(pairs):
    shape = _count_fill_shape(pairs)
    if shape is None:
        return False
    for inp, out in pairs:
        res = _apply_count_fill(inp, shape)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_count_fill(pairs, test_input):
    return _apply_count_fill(test_input, _count_fill_shape(pairs))


# --- Marker line: draw a straight line between the two cells of the marker
# --- color, rewriting crossed cells through a learned value mapping ---

def _line_endpoints(grid):
    cands = []
    for color in sorted(_unique_colors(grid)):
        cells = np.argwhere(grid == color)
        if len(cells) != 2:
            continue
        (r0, c0), (r1, c1) = cells.tolist()
        if r0 == r1 or c0 == c1 or abs(r0 - r1) == abs(c0 - c1):
            cands.append((color, (r0, c0), (r1, c1)))
    return cands[0] if len(cands) == 1 else None

def _apply_marker_line(grid, mapping):
    ep = _line_endpoints(grid)
    if ep is None:
        return None
    _color, (r0, c0), (r1, c1) = ep
    dr, dc = int(np.sign(r1 - r0)), int(np.sign(c1 - c0))
    res = grid.copy()
    r, c = r0 + dr, c0 + dc
    while (r, c) != (r1, c1):
        v = int(grid[r, c])
        if v not in mapping:
            return None
        res[r, c] = mapping[v]
        r += dr
        c += dc
    return res

def _marker_line_params(pairs):
    mapping = {}
    for inp, out in pairs:
        ep = _line_endpoints(inp)
        if ep is None:
            return None
        _color, (r0, c0), (r1, c1) = ep
        dr, dc = int(np.sign(r1 - r0)), int(np.sign(c1 - c0))
        r, c = r0 + dr, c0 + dc
        while (r, c) != (r1, c1):
            v, w = int(inp[r, c]), int(out[r, c])
            if v in mapping and mapping[v] != w:
                return None
            mapping[v] = w
            r += dr
            c += dc
    if not mapping:
        return None
    for inp, out in pairs:
        res = _apply_marker_line(inp, mapping)
        if res is None or not np.array_equal(res, out):
            return None
    return mapping

def _detect_marker_line(pairs):
    return _marker_line_params(pairs) is not None

def _solve_marker_line(pairs, test_input):
    return _apply_marker_line(test_input, _marker_line_params(pairs))


# --- Bbox crop: crop to the shape's bounding box, optionally swapping the
# --- two colors inside it ---

def _apply_bbox_crop(grid, swap):
    cells = np.argwhere(grid != 0)
    if len(cells) == 0:
        return None
    r0, r1 = cells[:, 0].min(), cells[:, 0].max()
    c0, c1 = cells[:, 1].min(), cells[:, 1].max()
    res = grid[r0:r1 + 1, c0:c1 + 1].copy()
    if swap:
        cols = sorted(_unique_colors(res))
        if len(cols) != 2:
            return None
        a, b = cols
        ma, mb = res == a, res == b
        res[ma], res[mb] = b, a
    return res

def _bbox_crop_swap(pairs):
    for swap in (False, True):
        if all((res := _apply_bbox_crop(inp, swap)) is not None
               and res.shape == out.shape and np.array_equal(res, out)
               for inp, out in pairs):
            return swap
    return None

def _detect_bbox_crop(pairs):
    return _bbox_crop_swap(pairs) is not None

def _solve_bbox_crop(pairs, test_input):
    return _apply_bbox_crop(test_input, _bbox_crop_swap(pairs))


# --- Spiral: an empty grid is filled with an inward clockwise spiral ---

def _apply_spiral(grid, color):
    if grid.any():
        return None
    H, W = grid.shape
    res = np.zeros_like(grid)
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    inb = lambda rr, cc: 0 <= rr < H and 0 <= cc < W

    def movable(r, c, d):
        # step ahead must be free; the cell after it must not already be drawn
        # (that one-cell gap is what forms the spiral's corridors)
        dr, dc = dirs[d]
        nr, nc = r + dr, c + dc
        n2r, n2c = r + 2 * dr, c + 2 * dc
        return (inb(nr, nc) and res[nr, nc] == 0
                and (not inb(n2r, n2c) or res[n2r, n2c] == 0))

    r = c = d = 0
    res[0, 0] = color
    while True:
        if not movable(r, c, d):
            d = (d + 1) % 4  # turn clockwise
            if not movable(r, c, d):
                break
        r, c = r + dirs[d][0], c + dirs[d][1]
        res[r, c] = color
    return res

def _spiral_color(pairs):
    cols = set()
    for _inp, out in pairs:
        cols |= _unique_colors(out)
    return cols.pop() if len(cols) == 1 else None

def _detect_spiral(pairs):
    color = _spiral_color(pairs)
    if color is None:
        return False
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_spiral(inp, color)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_spiral(pairs, test_input):
    return _apply_spiral(test_input, _spiral_color(pairs))


# --- Border sort: interior dots slide to hug the border matching their
# --- color; dots matching no border are removed ---

def _apply_border_sort(grid):
    H, W = grid.shape
    if H < 4 or W < 4:
        return None
    if grid[0, 0] or grid[0, W - 1] or grid[H - 1, 0] or grid[H - 1, W - 1]:
        return None

    def uni(vals):
        s = set(vals.tolist())
        return s.pop() if len(s) == 1 and 0 not in s else None

    T, B = uni(grid[0, 1:W - 1]), uni(grid[H - 1, 1:W - 1])
    L, R = uni(grid[1:H - 1, 0]), uni(grid[1:H - 1, W - 1])
    if None in (T, B, L, R) or len({T, B, L, R}) != 4:
        return None
    res = grid.copy()
    res[1:H - 1, 1:W - 1] = 0
    for r, c in np.argwhere(grid[1:H - 1, 1:W - 1] != 0) + 1:
        v = grid[r, c]
        if v == T:
            res[1, c] = v
        elif v == B:
            res[H - 2, c] = v
        elif v == L:
            res[r, 1] = v
        elif v == R:
            res[r, W - 2] = v
    return res

def _detect_border_sort(pairs):
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_border_sort(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_border_sort(pairs, test_input):
    return _apply_border_sort(test_input)


# --- Half mirror: one half of the grid is empty; fill it with the mirror
# --- image of the occupied half ---

def _apply_half_mirror(g):
    H, W = g.shape
    if H % 2 == 0:
        top, bot = g[:H // 2], g[H // 2:]
        if not top.any() and bot.any():
            return np.vstack([np.flipud(bot), bot])
        if not bot.any() and top.any():
            return np.vstack([top, np.flipud(top)])
    if W % 2 == 0:
        left, right = g[:, :W // 2], g[:, W // 2:]
        if not left.any() and right.any():
            return np.hstack([np.fliplr(right), right])
        if not right.any() and left.any():
            return np.hstack([left, np.fliplr(left)])
    return None

def _detect_half_mirror(pairs):
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_half_mirror(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_half_mirror(pairs, test_input):
    return _apply_half_mirror(test_input)


# --- Legend crop: crop the framed rectangle; scattered 2-cell "legend"
# --- dominoes (a,b) define the recoloring b -> a inside it ---

def _apply_legend_crop(grid):
    # the frame is the largest connected component; everything outside its
    # bounding box is legend dominoes or noise
    comps = _components(grid, diag=True)
    if not comps:
        return None
    _fc, cells = max(comps, key=lambda kv: len(kv[1]))
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    region = grid[r0:r1 + 1, c0:c1 + 1]
    # legend dominoes: horizontal pairs of two non-zero cells outside the box
    mapping = {}
    H, W = grid.shape
    inside = lambda r, c: r0 <= r <= r1 and c0 <= c <= c1
    for r in range(H):
        c = 0
        while c < W - 1:
            if (grid[r, c] != 0 and grid[r, c + 1] != 0
                    and not inside(r, c) and not inside(r, c + 1)
                    and (c == 0 or grid[r, c - 1] == 0)
                    and (c + 2 == W or grid[r, c + 2] == 0)):
                a, b = int(grid[r, c]), int(grid[r, c + 1])
                if b in mapping and mapping[b] != a:
                    return None
                mapping[b] = a
                c += 2
            else:
                c += 1
    if not mapping:
        return None
    res = region.copy()
    for b, a in mapping.items():
        res[region == b] = a
    return res

def _detect_legend_crop(pairs):
    for inp, out in pairs:
        res = _apply_legend_crop(inp)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_legend_crop(pairs, test_input):
    return _apply_legend_crop(test_input)


# --- Panel complement: two panels around a separator; if the second panel's
# --- cells exactly fill the first panel's holes, merge them, else keep first ---

def _apply_panel_complement(grid):
    segs = _split_panels(grid)
    if not segs or len(segs) != 2:
        return None
    a, b = segs
    holes = a == 0
    filled = b != 0
    if np.array_equal(holes, filled):
        res = a.copy()
        res[holes] = b[holes]
        return res
    return a.copy()

def _detect_panel_complement(pairs):
    for inp, out in pairs:
        res = _apply_panel_complement(inp)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_panel_complement(pairs, test_input):
    return _apply_panel_complement(test_input)


# --- Loop lining: closed loops get an inner lining and an outer halo in two
# --- learned colors (8-adjacent to the walls); open shapes stay untouched ---

def _apply_loop_lining(grid, shape, inner_c, outer_c):
    H, W = grid.shape
    reached = _flood_from_border(grid, grid != shape)
    res = grid.copy()
    for color, cells in _components(grid):
        if color != shape:
            continue
        cellset = set(cells)
        near = set()
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < H and 0 <= nc < W
                            and grid[nr, nc] == 0 and (nr, nc) not in cellset):
                        near.add((nr, nc))
        enclosed = {p for p in near if not reached[p]}
        if not enclosed:  # open shape: no interior, leave as-is
            continue
        for r, c in near:
            res[r, c] = inner_c if not reached[r, c] else outer_c
    return res

def _loop_lining_params(pairs):
    params = None
    for inp, out in pairs:
        ic = _unique_colors(inp)
        if len(ic) != 1:
            return None
        shape = ic.pop()
        new = _unique_colors(out) - {shape}
        if not new:
            continue
        if len(new) != 2:
            return None
        reached = _flood_from_border(inp, inp != shape)
        inner_c = outer_c = None
        for r, c in np.argwhere((inp == 0) & (out != 0)):
            if reached[r, c]:
                outer_c = int(out[r, c])
            else:
                inner_c = int(out[r, c])
        if inner_c is None or outer_c is None:
            return None
        cur = (shape, inner_c, outer_c)
        if params is None:
            params = cur
        elif params != cur:
            return None
    if params is None:
        return None
    for inp, out in pairs:
        if not np.array_equal(_apply_loop_lining(inp, *params), out):
            return None
    return params

def _detect_loop_lining(pairs):
    return _loop_lining_params(pairs) is not None

def _solve_loop_lining(pairs, test_input):
    return _apply_loop_lining(test_input, *_loop_lining_params(pairs))


# --- Dot path: connect the two single-cell markers with a path; a straight
# --- leg leaves the "from" marker, a diagonal leg arrives at the "to" one ---

def _dot_path_cells(grid, frm, to):
    pf, pt = np.argwhere(grid == frm), np.argwhere(grid == to)
    if len(pf) != 1 or len(pt) != 1:
        return None
    (r0, c0), (r1, c1) = pf[0].tolist(), pt[0].tolist()
    dr, dc = int(np.sign(r1 - r0)), int(np.sign(c1 - c0))
    ar, ac = abs(r1 - r0), abs(c1 - c0)
    if dr == 0 or dc == 0:
        return None
    cells = []
    if ar >= ac:
        # straight leg runs vertically in the column one step toward `to`;
        # the diagonal leg walks back from just before `to`
        for i in range(1, ar - ac + 1):
            cells.append((r0 + i * dr, c0 + dc))
        for i in range(1, ac):
            cells.append((r1 - i * dr, c1 - i * dc))
    else:
        for i in range(1, ac - ar + 1):
            cells.append((r0 + dr, c0 + i * dc))
        for i in range(1, ar):
            cells.append((r1 - i * dr, c1 - i * dc))
    return cells

def _dot_path_params(pairs):
    for frm_idx in (0, 1):
        params = None
        ok = True
        for inp, out in pairs:
            ic = sorted(_unique_colors(inp))
            if len(ic) != 2 or len(np.argwhere(inp != 0)) != 2:
                return None
            frm, to = (ic[0], ic[1]) if frm_idx == 0 else (ic[1], ic[0])
            new = _unique_colors(out) - set(ic)
            if len(new) != 1:
                return None
            path_c = new.pop()
            cur = (frm, to, path_c)
            if params is None:
                params = cur
            elif params != cur:
                ok = False
                break
        if not ok or params is None:
            continue
        if all(_apply_dot_path(inp, *params) is not None
               and np.array_equal(_apply_dot_path(inp, *params), out)
               for inp, out in pairs):
            return params
    return None

def _apply_dot_path(grid, frm, to, path_c):
    cells = _dot_path_cells(grid, frm, to)
    if cells is None:
        return None
    H, W = grid.shape
    res = grid.copy()
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W) or res[r, c] != 0:
            return None
        res[r, c] = path_c
    return res

def _detect_dot_path(pairs):
    return _dot_path_params(pairs) is not None

def _solve_dot_path(pairs, test_input):
    return _apply_dot_path(test_input, *_dot_path_params(pairs))


# --- Connect pairs: shapes whose centers share a row/column get joined by a
# --- line (in a learned color) through the gap between them ---

def _connect_pairs_color(pairs):
    new = set()
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        new |= _unique_colors(out) - _unique_colors(inp)
    return new.pop() if len(new) == 1 else None

def _apply_connect_pairs(grid, line_c):
    comps = _components(grid, diag=True)
    if len(comps) < 2:
        return None
    boxes = []
    for _color, cells in comps:
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        boxes.append((min(rs), max(rs), min(cs), max(cs)))
    res = grid.copy()
    drew = False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ar0, ar1, ac0, ac1 = boxes[i]
            br0, br1, bc0, bc1 = boxes[j]
            if (ar0 + ar1) == (br0 + br1) and (ac1 < bc0 or bc1 < ac0):
                r = (ar0 + ar1) // 2
                lo, hi = (ac1, bc0) if ac1 < bc0 else (bc1, ac0)
                # skip if a third shape sits in the gap (connect neighbors only)
                if any(k not in (i, j) and kr0 <= r <= kr1
                       and kc0 < hi and kc1 > lo
                       for k, (kr0, kr1, kc0, kc1) in enumerate(boxes)):
                    continue
                for c in range(lo + 1, hi):
                    if grid[r, c] == 0:
                        res[r, c] = line_c
                        drew = True
            elif (ac0 + ac1) == (bc0 + bc1) and (ar1 < br0 or br1 < ar0):
                c = (ac0 + ac1) // 2
                lo, hi = (ar1, br0) if ar1 < br0 else (br1, ar0)
                if any(k not in (i, j) and kc0 <= c <= kc1
                       and kr0 < hi and kr1 > lo
                       for k, (kr0, kr1, kc0, kc1) in enumerate(boxes)):
                    continue
                for r in range(lo + 1, hi):
                    if grid[r, c] == 0:
                        res[r, c] = line_c
                        drew = True
    return res if drew else None

def _detect_connect_pairs(pairs):
    line_c = _connect_pairs_color(pairs)
    if line_c is None:
        return False
    for inp, out in pairs:
        res = _apply_connect_pairs(inp, line_c)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_connect_pairs(pairs, test_input):
    return _apply_connect_pairs(test_input, _connect_pairs_color(pairs))


# --- Majority fill: each closed container is flooded with the majority color
# --- of the noise dots inside it; all other noise is erased ---

def _apply_majority_fill(grid):
    H, W = grid.shape
    comps = _components(grid, diag=True)
    if not comps:
        return None
    # container components are large; noise dots are the small remainder
    walls = [(color, cells) for color, cells in comps if len(cells) >= 6]
    if not walls:
        return None
    wall_mask = np.zeros((H, W), dtype=bool)
    for _color, cells in walls:
        for r, c in cells:
            wall_mask[r, c] = True
    reached = _flood_from_border(grid, ~wall_mask)
    enclosed = ~wall_mask & ~reached
    res = np.zeros_like(grid)
    res[wall_mask] = grid[wall_mask]
    if not enclosed.any():
        return None
    # group enclosed cells into per-container regions
    seen = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            if enclosed[i, j] and not seen[i, j]:
                stack, region = [(i, j)], []
                seen[i, j] = True
                while stack:
                    r, c = stack.pop()
                    region.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and enclosed[nr, nc] and not seen[nr, nc]):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                counts = {}
                for r, c in region:
                    v = int(grid[r, c])
                    if v:
                        counts[v] = counts.get(v, 0) + 1
                if not counts:
                    return None
                top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                if len(top) > 1 and top[0][1] == top[1][1]:
                    return None  # ambiguous majority
                for r, c in region:
                    res[r, c] = top[0][0]
    return res

def _detect_majority_fill(pairs):
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_majority_fill(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_majority_fill(pairs, test_input):
    return _apply_majority_fill(test_input)


# --- Box symmetrize: dots inside each box get completed to be symmetric
# --- across both of the box's center axes (union of the 4 reflections) ---

def _apply_box_symmetrize(grid):
    ic = sorted(_unique_colors(grid))
    if len(ic) != 2:
        return None
    for box in ic:
        content = ic[0] if box == ic[1] else ic[1]
        bboxes = []
        for color, cells in _components(grid, diag=True):
            if color != box:
                continue
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            bboxes.append((min(rs), max(rs), min(cs), max(cs)))
        ccells = np.argwhere(grid == content).tolist()
        if not ccells or not bboxes:
            continue
        assign = []
        ok = True
        for r, c in ccells:
            hits = [b for b in bboxes if b[0] < r < b[1] and b[2] < c < b[3]]
            if len(hits) != 1:
                ok = False
                break
            assign.append((r, c, hits[0]))
        if not ok:
            continue
        res = grid.copy()
        for r, c, (r0, r1, c0, c1) in assign:
            for nr in (r, r0 + r1 - r):
                for nc in (c, c0 + c1 - c):
                    if res[nr, nc] == 0:
                        res[nr, nc] = content
        return res
    return None

def _detect_box_symmetrize(pairs):
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_box_symmetrize(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_box_symmetrize(pairs, test_input):
    return _apply_box_symmetrize(test_input)


# --- Separator mirror: shapes living in a lattice of separator lines are
# --- completed to be symmetric across every line they span ---

def _grid_lines(grid):
    """Lattice color plus the indices of its full uniform rows/columns."""
    H, W = grid.shape
    rows = [i for i in range(H)
            if grid[i, 0] != 0 and len(set(grid[i, :].tolist())) == 1]
    cols = [j for j in range(W)
            if grid[0, j] != 0 and len(set(grid[:, j].tolist())) == 1]
    colors = ({int(grid[i, 0]) for i in rows}
              | {int(grid[0, j]) for j in cols})
    if len(colors) != 1:
        return None
    L = colors.pop()
    return L, set(rows), set(cols)

def _apply_sep_mirror(grid):
    gl = _grid_lines(grid)
    if gl is None:
        return None
    L, srows, scols = gl
    H, W = grid.shape

    def nbrs(r, c):
        # 8-adjacency, hopping over separator lines so the halves of a shape
        # on either side of a line belong to one logical component
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                while (0 <= nr < H and 0 <= nc < W
                       and (nr in srows or nc in scols)):
                    nr += dr
                    nc += dc
                if 0 <= nr < H and 0 <= nc < W:
                    yield nr, nc

    seen = np.zeros((H, W), dtype=bool)
    res = grid.copy()
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 0 or grid[i, j] == L or seen[i, j]:
                continue
            color = grid[i, j]
            stack, comp = [(i, j)], set()
            seen[i, j] = True
            while stack:
                r, c = stack.pop()
                comp.add((r, c))
                for nr, nc in nbrs(r, c):
                    if grid[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            axes_r = [s for s in srows
                      if any(r < s for r, _ in comp) and any(r > s for r, _ in comp)]
            axes_c = [s for s in scols
                      if any(c < s for _, c in comp) and any(c > s for _, c in comp)]
            # closure under the spanned reflections
            cells = set(comp)
            frontier = set(comp)
            while frontier:
                new = set()
                for r, c in frontier:
                    for s in axes_r:
                        new.add((2 * s - r, c))
                    for s in axes_c:
                        new.add((r, 2 * s - c))
                new -= cells
                for r, c in new:
                    if not (0 <= r < H and 0 <= c < W):
                        return None
                    if grid[r, c] != 0 and grid[r, c] != color:
                        return None
                cells |= new
                frontier = new
            for r, c in cells:
                res[r, c] = color
    return res

def _detect_sep_mirror(pairs):
    changed = False
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_sep_mirror(inp)
        if res is None or not np.array_equal(res, out):
            return False
        if not np.array_equal(res, inp):
            changed = True
    return changed

def _solve_sep_mirror(pairs, test_input):
    return _apply_sep_mirror(test_input)


# --- Chevron: a lone marker in a 1-row grid expands into an NxN pattern of
# --- a V of marker cells plus trailing diagonals in a second color ---

def _chevron_color(pairs):
    S = set()
    for inp, out in pairs:
        S |= _unique_colors(out) - _unique_colors(inp)
    return S.pop() if len(S) == 1 else None

def _apply_chevron(grid, S):
    if grid.shape[0] != 1:
        return None
    nz = np.argwhere(grid != 0)
    if len(nz) != 1:
        return None
    c0 = int(nz[0][1])
    M = int(grid[0, c0])
    N = grid.shape[1]
    res = np.zeros((N, N), dtype=grid.dtype)
    for r in range(N):
        for c in range(N):
            d = abs(c - c0)
            if d == r:
                res[r, c] = M
            elif r > d and (c - r - c0) % 4 == 0:
                res[r, c] = S
    return res

def _detect_chevron(pairs):
    S = _chevron_color(pairs)
    if S is None:
        return False
    for inp, out in pairs:
        res = _apply_chevron(inp, S)
        if res is None or res.shape != out.shape or not np.array_equal(res, out):
            return False
    return True

def _solve_chevron(pairs, test_input):
    return _apply_chevron(test_input, _chevron_color(pairs))


# --- Gravity holes: shapes fall into the gaps of a floor structure, filling
# --- each gap bottom-up and stacking any overflow above it ---

def _apply_gravity(grid):
    H, W = grid.shape
    bottom = set(grid[H - 1, :].tolist())
    if len(bottom) != 1 or 0 in bottom:
        return None
    F = bottom.pop()
    frows = np.argwhere(grid == F)[:, 0]
    fr0 = int(frows.min())
    # holes: empty pockets inside the floor's rows, grouped 4-connected
    holes, seen = [], set()
    for r in range(fr0, H):
        for c in range(W):
            if grid[r, c] == 0 and (r, c) not in seen:
                stack, region = [(r, c)], []
                seen.add((r, c))
                while stack:
                    rr, cc = stack.pop()
                    region.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if (fr0 <= nr < H and 0 <= nc < W
                                and grid[nr, nc] == 0 and (nr, nc) not in seen):
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                holes.append(region)
    shapes = [(color, cells) for color, cells in _components(grid, diag=True)
              if color != F and all(r < fr0 for r, c in cells)]
    if not holes or len(shapes) != len(holes):
        return None
    holes.sort(key=len)
    shapes.sort(key=lambda kv: len(kv[1]))
    res = grid.copy()
    for _color, cells in shapes:
        for r, c in cells:
            res[r, c] = 0
    for hole, (color, cells) in zip(holes, shapes):
        n = len(cells)
        if n < len(hole):
            return None
        for r, c in hole:
            res[r, c] = color
        placed = len(hole)
        cols = sorted({c for _r, c in hole})
        r = min(r for r, _c in hole) - 1
        while placed < n:
            if r < 0:
                return None
            for c in cols:
                if placed >= n:
                    break
                if res[r, c] != 0:
                    return None
                res[r, c] = color
                placed += 1
            r -= 1
    return res

def _detect_gravity(pairs):
    for inp, out in pairs:
        if inp.shape != out.shape:
            return False
        res = _apply_gravity(inp)
        if res is None or not np.array_equal(res, out):
            return False
    return True

def _solve_gravity(pairs, test_input):
    return _apply_gravity(test_input)


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        pairs = [(s.get_input_data().data(), s.get_output_data().data())
                 for s in arc_problem.training_set()]
        test_input = arc_problem.test_set().get_input_data().data()

        if _detect_geometric(pairs) is not None:
            return [_solve_geometric(pairs, test_input)]

        if _detect_color_swap(pairs) is not None:
            return [_solve_color_swap(pairs, test_input)]

        if _detect_color_inversion(pairs) is not None:
            result = _solve_color_inversion(pairs, test_input)
            if result is not None:
                return [result]

        if _detect_blockfill(pairs):
            return [_solve_blockfill(pairs, test_input)]

        if _detect_diagonal_x(pairs):
            return [_solve_diagonal_x(pairs, test_input)]

        if _detect_hollow(pairs):
            return [_solve_hollow(pairs, test_input)]

        if _detect_quad_mirror(pairs):
            return [_solve_quad_mirror(pairs, test_input)]

        if _detect_staircase(pairs):
            return [_solve_staircase(pairs, test_input)]

        if _detect_edge_fill(pairs) is not None:
            return [_solve_edge_fill(pairs, test_input)]

        if _detect_cornercrop(pairs):
            return [_solve_cornercrop(pairs, test_input)]

        if _detect_movemarker(pairs):
            return [_solve_movemarker(pairs, test_input)]

        if _detect_colsort(pairs):
            return [_solve_colsort(pairs, test_input)]

        if _detect_rooms(pairs):
            return [_solve_rooms(pairs, test_input)]

        if _detect_closedloop(pairs):
            return [_solve_closedloop(pairs, test_input)]

        if _detect_panel_overlay(pairs):
            return [_solve_panel_overlay(pairs, test_input)]

        if _detect_diag_rays(pairs):
            return [_solve_diag_rays(pairs, test_input)]

        if _detect_arrow(pairs):
            return [_solve_arrow(pairs, test_input)]

        if _detect_box_reflect(pairs):
            return [_solve_box_reflect(pairs, test_input)]

        if _detect_boxstamp(pairs):
            return [_solve_boxstamp(pairs, test_input)]

        if _detect_mirror_tile3(pairs):
            return [_solve_mirror_tile3(pairs, test_input)]

        if _detect_lattice_hist(pairs):
            return [_solve_lattice_hist(pairs, test_input)]

        if _detect_count_fill(pairs):
            return [_solve_count_fill(pairs, test_input)]

        if _detect_marker_line(pairs):
            return [_solve_marker_line(pairs, test_input)]

        if _detect_spiral(pairs):
            return [_solve_spiral(pairs, test_input)]

        if _detect_border_sort(pairs):
            return [_solve_border_sort(pairs, test_input)]

        if _detect_half_mirror(pairs):
            return [_solve_half_mirror(pairs, test_input)]

        if _detect_bbox_crop(pairs):
            return [_solve_bbox_crop(pairs, test_input)]

        if _detect_legend_crop(pairs):
            return [_solve_legend_crop(pairs, test_input)]

        if _detect_panel_complement(pairs):
            return [_solve_panel_complement(pairs, test_input)]

        if _detect_loop_lining(pairs):
            return [_solve_loop_lining(pairs, test_input)]

        if _detect_dot_path(pairs):
            return [_solve_dot_path(pairs, test_input)]

        if _detect_connect_pairs(pairs):
            return [_solve_connect_pairs(pairs, test_input)]

        if _detect_majority_fill(pairs):
            return [_solve_majority_fill(pairs, test_input)]

        if _detect_box_symmetrize(pairs):
            return [_solve_box_symmetrize(pairs, test_input)]

        if _detect_sep_mirror(pairs):
            return [_solve_sep_mirror(pairs, test_input)]

        if _detect_chevron(pairs):
            return [_solve_chevron(pairs, test_input)]

        if _detect_gravity(pairs):
            return [_solve_gravity(pairs, test_input)]

        logic_result = _solve_binary_logic(pairs, test_input)
        if logic_result is not None:
            return [logic_result]

        return []
