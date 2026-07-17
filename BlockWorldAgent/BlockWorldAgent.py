class BlockWorldAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, initial_arrangement, goal_arrangement):
        #This agent uses means-ends analysis. At each step it compares
        #the current state against the goal state and selects the move
        #that best reduces the difference between them:
        #
        # 1. If some clear block can be moved directly into its final
        #    goal position (its goal support is settled and clear, or
        #    its goal position is the bottom of a new stack on the
        #    table), make that constructive move.
        # 2. Otherwise, no block can reach its final position yet, so
        #    move a clear, unsettled block to the table to unbury the
        #    blocks beneath it (a setup move toward the subgoal).
        #
        #A block is "settled" when it sits exactly where the goal wants
        #it AND everything beneath it is settled too. Settled blocks are
        #never moved again, so every constructive move permanently
        #reduces the difference and the agent always terminates.

        current = [list(stack) for stack in initial_arrangement]

        #For each block, record what should be directly beneath it in
        #the goal state ("Table" if it is the bottom of a goal stack).
        goal_below = {}
        for stack in goal_arrangement:
            for i, block in enumerate(stack):
                goal_below[block] = stack[i - 1] if i > 0 else "Table"

        total_blocks = len(goal_below)
        moves = []

        def find_settled():
            #A stack's settled blocks are the prefix that already
            #matches a goal stack from the table up.
            settled = set()
            for stack in current:
                for i, block in enumerate(stack):
                    below = stack[i - 1] if i > 0 else "Table"
                    if goal_below[block] == below:
                        settled.add(block)
                    else:
                        break
            return settled

        def move_block(block, destination):
            #Apply a move to the current state and record it.
            for stack in current:
                if stack[-1] == block:
                    stack.pop()
                    if not stack:
                        current.remove(stack)
                    break
            if destination == "Table":
                current.append([block])
            else:
                for stack in current:
                    if stack[-1] == destination:
                        stack.append(block)
                        break
            moves.append((block, destination))

        while True:
            settled = find_settled()
            if len(settled) == total_blocks:
                break

            clear_blocks = [stack[-1] for stack in current]

            #Step 1: look for a constructive move — a clear block whose
            #goal position is ready to receive it.
            constructive = None
            for block in clear_blocks:
                if block in settled:
                    continue
                destination = goal_below[block]
                if destination == "Table":
                    constructive = (block, "Table")
                    break
                if destination in settled and destination in clear_blocks:
                    constructive = (block, destination)
                    break

            if constructive is not None:
                move_block(*constructive)
                continue

            #Step 2: no block can be placed in its final position, so
            #unstack a clear, unsettled block onto the table to unbury
            #the blocks we need. (Skip blocks already alone on the
            #table; moving them accomplishes nothing.)
            for stack in current:
                block = stack[-1]
                if block not in settled and len(stack) > 1:
                    move_block(block, "Table")
                    break

        return moves
