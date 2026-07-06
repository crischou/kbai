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
    for orient in ("row", "col"):
        splits = []
        ok = True
        for inp, out in training_pairs:
            sep = _find_separator(inp, orient)
            if sep is None:
                ok = False
                break
            a, b = _split(inp, sep, orient)
            if a.shape != b.shape or a.shape != out.shape:
                ok = False
                break
            splits.append((a, b, out))
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
                    sep = _find_separator(test_input, orient)
                    if sep is None:
                        return None
                    a, b = _split(test_input, sep, orient)
                    if a.shape != b.shape:
                        return None
                    return _logic_build(a, b, op, mode, flat_color)
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

        logic_result = _solve_binary_logic(pairs, test_input)
        if logic_result is not None:
            return [logic_result]

        return []
