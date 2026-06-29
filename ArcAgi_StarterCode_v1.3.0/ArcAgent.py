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



def _find_col_separator(grid):
    """Find a full column of a single non-zero value."""
    for col in range(grid.shape[1]):
        col_vals = grid[:, col]
        unique = set(col_vals.tolist())
        if len(unique) == 1 and 0 not in unique:
            return col, int(col_vals[0])
    return None, None

def _find_row_separator(grid):
    """Find a full row of a single non-zero value."""
    for row in range(grid.shape[0]):
        row_vals = grid[row, :]
        unique = set(row_vals.tolist())
        if len(unique) == 1 and 0 not in unique:
            return row, int(row_vals[0])
    return None, None

def _detect_and(training_pairs):
    for inp, out in training_pairs:
        sep, _ = _find_col_separator(inp)
        if sep is None:
            return False
        left, right = inp[:, :sep], inp[:, sep+1:]
        if left.shape != right.shape or left.shape != out.shape:
            return False
        if not np.array_equal((left != 0) & (right != 0), out != 0):
            return False
    return True

def _solve_and(training_pairs, test_input):
    sep, _ = _find_col_separator(test_input)
    left, right = test_input[:, :sep], test_input[:, sep+1:]
    out_color = _unique_colors(training_pairs[0][1]).pop()
    result = np.zeros_like(left)
    result[(left != 0) & (right != 0)] = out_color
    return result

def _detect_or(training_pairs):
    for inp, out in training_pairs:
        sep, _ = _find_row_separator(inp)
        if sep is None:
            return False
        top, bot = inp[:sep, :], inp[sep+1:, :]
        if top.shape != out.shape or bot.shape != out.shape:
            return False
        if not np.array_equal((top != 0) | (bot != 0), out != 0):
            return False
    return True

def _solve_or(training_pairs, test_input):
    sep, _ = _find_row_separator(test_input)
    top, bot = test_input[:sep, :], test_input[sep+1:, :]
    out_color = _unique_colors(training_pairs[0][1]).pop()
    result = np.zeros_like(top)
    result[(top != 0) | (bot != 0)] = out_color
    return result

def _detect_nor(training_pairs):
    for inp, out in training_pairs:
        sep, _ = _find_col_separator(inp)
        if sep is None:
            return False
        left, right = inp[:, :sep], inp[:, sep+1:]
        if left.shape != out.shape or right.shape != out.shape:
            return False
        if not np.array_equal((left == 0) & (right == 0), out != 0):
            return False
    return True

def _solve_nor(training_pairs, test_input):
    sep, _ = _find_col_separator(test_input)
    left, right = test_input[:, :sep], test_input[:, sep+1:]
    out_color = _unique_colors(training_pairs[0][1]).pop()
    result = np.zeros_like(left)
    result[(left == 0) & (right == 0)] = out_color
    return result

def _detect_rotation(training_pairs):
    """Returns k if np.rot90(input, k) matches output for all pairs."""
    for k in [1, 2, 3]:
        if all(np.array_equal(np.rot90(inp, k), out) for inp, out in training_pairs):
            return k
    return None

def _solve_rotation(training_pairs, test_input):
    k = _detect_rotation(training_pairs)
    return np.rot90(test_input, k)

class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        pairs = [(s.get_input_data().data(), s.get_output_data().data())
                 for s in arc_problem.training_set()]
        test_input = arc_problem.test_set().get_input_data().data()

        if _detect_rotation(pairs) is not None:
            return [_solve_rotation(pairs, test_input)]

        if _detect_color_swap(pairs) is not None:
            return [_solve_color_swap(pairs, test_input)]

        if _detect_color_inversion(pairs) is not None:
            result = _solve_color_inversion(pairs, test_input)
            if result is not None:
                return [result]

        if _detect_and(pairs):
            return [_solve_and(pairs, test_input)]

        if _detect_or(pairs):
            return [_solve_or(pairs, test_input)]

        if _detect_nor(pairs):
            return [_solve_nor(pairs, test_input)]

        return []
