from collections import deque


class SemanticNetsAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, initial_sheep, initial_wolves):
        #Add your code here! Your solve method should receive
        #the initial number of sheep and wolves as integers,
        #and return a list of 2-tuples that represent the moves
        #required to get all sheep and wolves from the left
        #side of the river to the right.
        #
        #If it is impossible to move the animals over according
        #to the rules of the problem, return an empty list of
        #moves.

        total_sheep = initial_sheep
        total_wolves = initial_wolves

        # State: (sheep_left, wolves_left, boat_side)
        #   boat_side: 0 = boat on left bank, 1 = boat on right bank
        # Goal: everything on the right with the boat on the right.
        start = (total_sheep, total_wolves, 0)
        goal = (0, 0, 1)

        # Possible boat loads (sheep, wolves): 1 or 2 animals total.
        boat_loads = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

        def is_valid(sheep_left, wolves_left):
            sheep_right = total_sheep - sheep_left
            wolves_right = total_wolves - wolves_left

            # Counts must stay within bounds.
            if sheep_left < 0 or wolves_left < 0:
                return False
            if sheep_right < 0 or wolves_right < 0:
                return False

            # Sheep are eaten only if wolves outnumber them on a bank
            # that has at least one sheep.
            if sheep_left > 0 and wolves_left > sheep_left:
                return False
            if sheep_right > 0 and wolves_right > sheep_right:
                return False

            return True

        # BFS guarantees the first solution found uses the fewest moves.
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (sheep_left, wolves_left, boat_side), path = queue.popleft()

            if (sheep_left, wolves_left, boat_side) == goal:
                return path

            for sheep_move, wolves_move in boat_loads:
                if boat_side == 0:
                    # Boat travels left -> right: remove from left bank.
                    new_sheep = sheep_left - sheep_move
                    new_wolves = wolves_left - wolves_move
                    new_side = 1
                else:
                    # Boat travels right -> left: add to left bank.
                    new_sheep = sheep_left + sheep_move
                    new_wolves = wolves_left + wolves_move
                    new_side = 0

                new_state = (new_sheep, new_wolves, new_side)

                if not is_valid(new_sheep, new_wolves):
                    continue
                if new_state in visited:
                    continue

                visited.add(new_state)
                queue.append((new_state, path + [(sheep_move, wolves_move)]))

        # No path to the goal -> unsolvable.
        return []

