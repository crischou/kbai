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

# Connected components of equal-colored, 4-adjacent non-zero cells.
def _components(grid):
    H, W = grid.shape
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
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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

        logic_result = _solve_binary_logic(pairs, test_input)
        if logic_result is not None:
            return [logic_result]

        return []
