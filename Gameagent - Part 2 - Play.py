# Dynamic Connect 3 - Game Agent Multi Player

# --------------------------
# Need a Depth-Cutoff heuristic.
# You have to be able to paly either white (first move, then you receive responses) or black (you respond after the 1st move). Are you white or black = depends on when you connect to the server (gae=me 37 black)
# Board tracking = need to have a way to store game state - which data structure to implement 
# Draw test = need to have a mechanism for draws - three repeated identical game states 
# Displays board state. 
# --------------------------

# Part I = 
#---------------------------
# minimax algorithm to search tree. Assign large negative & positive values to terminal nodes. 
# naive heuristic = h(n) = num 2 in a row positions runs white - num 2 in a row positions runs black 
# Then change minimax to alpha beta for report. 
#---------------------------

# Game board State, Actions, Transitions, Terminal Tests (Win, Lose, Draw), Rewards

# Make the 2D board into a 1d bitboard indexes. 

class GameState():
    '''
    Data structure = bitboard
    * Need to convert 2d board state into the bitboard : need to have a index feature
    * Use of white_bits, black_bits = for white and black positions. Union for Occupited positions. 

    For search implementation = 
    * Need TT table so that we don't recompute previously seen trees
    * Need a make / unmake feature so that we don't need to recompute entire trees every time. 
    '''

    def __init__(self, board_width, board_height, white_bitboard, black_bitboard, turn_to_move):
        self.board_width = board_width
        self.board_height = board_height
        self.white_bitboard = white_bitboard
        self.black_bitboard = black_bitboard
        self.turn_to_move = turn_to_move  
        self.board_mask = self._board_mask(board_width, board_height)
        self.neighbours = self._precompute_neighbours()
        self.terminal_states = self._precompute_terminal_states()
        self.initial_state(board_width, board_height)
        

    @staticmethod
    def _board_mask(board_width, board_height):
        # necessary to be able to implement bitboards (integers in python have exploding bits, but you want to stay in bounds)
        # defines the universe that represents each bit on the game board 
        if board_width <= 0 or board_height <= 0:
            raise ValueError("board_width and board_height must be positive integers.")
        num_cells = board_width * board_height
        return (1 << num_cells) - 1
        
    def inbounds(self, x, y):
        if (1 <= x <= self.board_width) and (1 <= y <= self.board_height):
            return True
        else:
            return False

    def directions_moves(self):
        # assigns a move to each direction N, S, W, E
        return {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

    def to_index(self, x, y):
        # translate 2D board game position to bitboard index
        return (y - 1) * (self.board_width) + (x - 1)

    def _precompute_neighbours(self):
        # used to check for possible moves
        neighbours_list = [None] * (self.board_width * self.board_height)
        directions = self.directions_moves()
        for y in range(1, self.board_height + 1):     
            for x in range(1, self.board_width + 1):
                index = self.to_index(x, y)
                direction_dict = {}
                for direction, (dx, dy) in directions.items():
                    next_x, next_y = x + dx, y + dy
                    if self.inbounds(next_x, next_y):
                        direction_dict[direction] = self.to_index(next_x, next_y)
                neighbours_list[index] = direction_dict
        return neighbours_list

    def _precompute_terminal_states(self): 
        # uses to check for terminal states
        masks = []
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # E, S, SE, NE
        for y in range(1, self.board_height + 1):
            for x in range(1, self.board_width + 1):
                for dx, dy in directions:
                    is_valid = True
                    cells = []
                    for step in range(3):
                        next_x = x + step * dx
                        next_y = y + step * dy
                        if not self.inbounds(next_x, next_y):
                            is_valid = False
                            break
                        cells.append((next_x, next_y))
                    if not is_valid:
                        continue
                    mask = 0
                    for (cell_x, cell_y) in cells:
                        mask |= 1 << self.to_index(cell_x, cell_y)
                    masks.append(mask)
        return masks

    def initial_state(self, board_width, board_height):
        # the initial board state
        white_start_positions = [(1,1), (5,2), (1,3), (5,4)]
        black_start_positions = [(5,1), (1,2), (5,3), (1,4)]

        if (self.board_width, self.board_height) == (7, 6):
            white_start_positions = [(x+1, y+1) for (x,y) in white_start_positions]
            black_start_positions = [(x+1, y+1) for (x,y) in black_start_positions]
        elif (self.board_width, self.board_height) != (5, 4):
            raise ValueError("Unsupported board size")

        white_bitboard, black_bitboard = 0, 0
        for (x, y) in white_start_positions:
            white_bitboard |= 1 << self.to_index(x, y)
        for (x, y) in black_start_positions:
            black_bitboard |= 1 << self.to_index(x, y)

        self.white_bitboard = white_bitboard & self.board_mask
        self.black_bitboard = black_bitboard & self.board_mask
        self.turn_to_move = 0
        assert (self.white_bitboard & self.black_bitboard) == 0
        assert ((self.white_bitboard | self.black_bitboard) & ~self.board_mask) == 0
    
    def tt_key(self):
        # return a tuple that uniquely identifies the full board position
        return (self.white_bitboard, self.black_bitboard, self.turn_to_move, self.board_width, self.board_height)

    def allowable_moves(self, x, y): 
        # computes the allowable moves from a x,y position on the board
        index = self.to_index(x, y)
        if self.turn_to_move == 0:
            current_player_bitboard = self.white_bitboard
        else:
            current_player_bitboard = self.black_bitboard
        if ((current_player_bitboard >> index) & 1) == 0:
            return {}
        occupancy_bitboard = (self.white_bitboard | self.black_bitboard) & self.board_mask
        allowed_moves = {}
        for direction, destination_index in self.neighbours[index].items():
            if ((occupancy_bitboard >> destination_index) & 1) == 0:
                allowed_moves[direction] = destination_index
        return allowed_moves
    
    def legal_moves(self):
        # computes all legal_moves from the positions of your pieces
        moves = []
        current_player_bitboard = self.white_bitboard if self.turn_to_move == 0 else self.black_bitboard
        num_cells = self.board_width * self.board_height
        for index in range(num_cells):
            if ((current_player_bitboard >> index) & 1) == 0:
                continue
            x = (index % self.board_width) + 1
            y = (index // self.board_width) + 1
            allowed = self.allowable_moves(x, y)
            for destination_index in allowed.values():
                moves.append((index, destination_index))
        return moves

    def terminal_test(self):
        # tests if board is in a terminal state 
        INF = float("inf")
        for terminal_mask in self.terminal_states:
            if (self.white_bitboard & terminal_mask) == terminal_mask:
                return +INF
            if (self.black_bitboard & terminal_mask) == terminal_mask:
                return -INF
        return None
    
    def terminal_value(self):
        return self.terminal_test()

# make and unmake are tools used by the search engine in order to explore the tree in place without having to allocate fresh states to each child.

    def make(self, move): 
        source_index, destination_index = move
        if destination_index not in self.neighbours[source_index].values():
            raise ValueError("destination is not a 1-step neighbour")
        occupancy = (self.white_bitboard | self.black_bitboard) & self.board_mask
        if ((occupancy >> destination_index) & 1) != 0:
            raise ValueError("destination is occupied")
        if self.turn_to_move == 0:
            if ((self.white_bitboard >> source_index) & 1) == 0:
                raise ValueError("no white piece at source")
            self.white_bitboard ^= (1 << source_index) | (1 << destination_index)
        else:
            if ((self.black_bitboard >> source_index) & 1) == 0:
                raise ValueError("no black piece at source")
            self.black_bitboard ^= (1 << source_index) | (1 << destination_index)
        self.turn_to_move ^= 1
        return (source_index, destination_index)

    def unmake(self, undo): 
        source_index, destination_index = undo
        self.turn_to_move ^= 1
        if self.turn_to_move == 0:
            self.white_bitboard ^= (1 << source_index) | (1 << destination_index)
        else:
            self.black_bitboard ^= (1 << source_index) | (1 << destination_index)

    def decode_moves(self, move_string):
        # convert x,y{N,S,E,W} move from opponent into a input for state transition. 
        move_str = move_string.strip().upper().replace(" ", "")
        if len(move_str) < 3:
            raise ValueError("Invalid move format.")
        x = int(move_str[0])
        y = int(move_str[1])
        direction = move_str[2]
        if not self.inbounds(x, y):
            raise ValueError("Out of bounds.")
        source_index = self.to_index(x, y)
        if direction not in self.neighbours[source_index]:
            raise ValueError("Direction is not valid.")
        destination_index = self.neighbours[source_index][direction]
        return (source_index, destination_index)

    def encode_moves(self, move):
        # convert bitboard move into a x,y{N,S,E,W} move to send to opponent
        source_index, destination_index = move
        x1 = (source_index % self.board_width) + 1
        y1 = (source_index // self.board_width) + 1
        x2 = (destination_index % self.board_width) + 1
        y2 = (destination_index // self.board_width) + 1
        dx = x2 - x1
        dy = y2 - y1
        if (dx, dy) == (1, 0):
            direction = "E"
        elif (dx, dy) == (-1, 0):
            direction = "W"
        elif (dx, dy) == (0, 1):
            direction = "S"
        elif (dx, dy) == (0, -1):
            direction = "N"
        else:
            raise ValueError("Incorrect move.")
        return f"{x1}{y1}{direction}"

    def display(self): 
        # displays the board game at every move
        for y in range(1, self.board_height + 1):
            cells = []
            for x in range(1, self.board_width + 1):
                index = self.to_index(x, y)
                if ((self.white_bitboard >> index) & 1) == 1:
                    cells.append("0")
                elif ((self.black_bitboard >> index) & 1) == 1:
                    cells.append("1")
                else:
                    cells.append(" ")
            print(" , ".join(cells))



# so the GameState is finished. Now we want to be able to use the different GameState tools by our Search Class in order to build out search trees and return the best  moves. 
# how does the search algorithm incorporate the game state into its search ? 
# need a transposition table to implement a draw test. 
# NEED DISPLAY OF GAME BOARD AT EACH MOVE. 

# Part I run minimax algorithm. graph depth vs time complexity etc to figure out a correct depth cut off. Implement the heuristic for that depth. 

# THEN : run the the same thing but with alpha beta pruning & heuristic at depth cut off. 
# Graph what you need for report I

# --------------------------------------------------------
# How to track depths of the search and have a cut off that then heuristic_evaluations node with the naive heuristic. 
# Test different depths and graph time to complete search for each depth : minimax then alpha beta pruning. 


import time
from math import inf
import socket

class Engine:
    def __init__(self):
        self.nodes = 0
        self._mask_cache = {}
        self.transposition_table = {}
        # counts of real-game occurrences for threefold-repetition detection
        self.transposition_table_counts = {}
        self.use_transposition_table = False
        self.move_ordering = "neutral"
        self.history_table = {}

#  naive heuristic first = number of 2 in a row positions for white - number of 2 in a row positions for black
#  divide search algorithm into search for value then selection of the best move implementing make unmake for efficient tree traversals and value 

    def heuristic_evaluation(self, state):
        line_masks = self._line_masks(state.board_width, state.board_height, 2)
        white_bits = state.white_bitboard
        black_bits = state.black_bitboard
        score = 0
        for mask in line_masks:
            if (white_bits & mask) == mask and (black_bits & mask) == 0:
                score += 1
            elif (black_bits & mask) == mask and (white_bits & mask) == 0:
                score -= 1
        return float(score)

    def minimax_value(self, state, depth):
        # check timer for iterative-deepening cutoff
        if getattr(self, 'stop_time', None) is not None and time.perf_counter() > self.stop_time:
            raise TimeoutError()
        self.nodes += 1

        if hasattr(state, "terminal_value"):
            terminal = state.terminal_value()
            if terminal is not None:
                return terminal
        if depth == 0:
            return self.heuristic_evaluation(state)

        key = state.tt_key()
        entry = self.transposition_table.get(key) if self.use_transposition_table else None
        tt_best = None
        if entry:
            stored_depth, flag, value, best_move = entry
            if stored_depth >= depth and flag == "EXACT":
                return value
            tt_best = best_move

        moves = state.legal_moves()
        if not moves:
            return 0.0
        # apply move ordering policy
        moves = self._order_moves(state, moves, tt_best, maximizing=(state.turn_to_move == 0))

        if state.turn_to_move == 0:
            best_value = -inf
            best_move = moves[0]
            for move in moves:
                undo_token = state.make(move)
                child_value = self.minimax_value(state, depth - 1)
                state.unmake(undo_token)
                if child_value > best_value:
                    best_value = child_value
                    best_move = move
            self.transposition_table[key] = (depth, "EXACT", best_value, best_move)
            return best_value
        else:
            best_value = +inf
            best_move = moves[0]
            for move in moves:
                undo_token = state.make(move)
                child_value = self.minimax_value(state, depth - 1)
                state.unmake(undo_token)
                if child_value < best_value:
                    best_value = child_value
                    best_move = move
            self.transposition_table[key] = (depth, "EXACT", best_value, best_move)
            return best_value
    
    def choose_move_minimax(self, state, depth):
        self.nodes = 0
        moves = state.legal_moves()
        if not moves:
            return None, 0.0, 0

        moves = self._order_moves(state, moves, self.transposition_table.get(state.tt_key())[3] if self.use_transposition_table and state.tt_key() in self.transposition_table else None, maximizing=(state.turn_to_move == 0))

        if state.turn_to_move == 0:
            best_value = -inf
            best_move = moves[0]
            for move in moves:
                undo_token = state.make(move)
                value = self.minimax_value(state, depth - 1)
                state.unmake(undo_token)
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_move, best_value, self.nodes
        else:
            best_value = +inf
            best_move = moves[0]
            for move in moves:
                undo_token = state.make(move)
                value = self.minimax_value(state, depth - 1)
                state.unmake(undo_token)
                if value < best_value:
                    best_value = value
                    best_move = move
            return best_move, best_value, self.nodes
  
# --------------------------------------
# Methods for Part I - Graphs = 

    def sweep_depths(self, state, max_depth=6, per_depth_budget_sec=25.0):
        results = []
        for depth in range(1, max_depth + 1):
            t_start = time.perf_counter()
            best_move, best_value, node_count = self.choose_move_minimax(state, depth)
            elapsed = time.perf_counter() - t_start
            nodes_per_sec = node_count / elapsed if elapsed > 0 else float("nan")
            pv_move = state.encode_moves(best_move) if best_move is not None else None
            results.append({
                'depth': depth,
                'value': best_value,
                'nodes': node_count,
                'elapsed': elapsed,
                'nodes_per_sec': nodes_per_sec,
                'pv_move': pv_move
            })
            if elapsed > per_depth_budget_sec:
                break
        return results

    def _line_masks(self, width, height, run_len):
        key = (width, height, run_len)
        if key in self._mask_cache:
            return self._mask_cache[key]

        masks = []
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        def in_bounds(x, y):
            return 1 <= x <= width and 1 <= y <= height

        def to_index(x, y):
            return (y - 1) * width + (x - 1)

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                for dx, dy in directions:
                    ok = True
                    mask = 0
                    for step in range(run_len):
                        nx = x + step * dx
                        ny = y + step * dy
                        if not in_bounds(nx, ny):
                            ok = False
                            break
                        mask |= 1 << to_index(nx, ny)
                    if ok:
                        masks.append(mask)

        self._mask_cache[key] = masks
        return masks

    def _score_move(self, state, move):
        # simple static move scorer: prefer moves that create 2-in-a-row for the side to move
        source_index, destination_index = move
        # make the move, evaluate heuristic, then unmake
        undo = state.make(move)
        score = self.heuristic_evaluation(state)
        state.unmake(undo)
        return score

    def _order_moves(self, state, moves, tt_best_move=None, maximizing=True):
        # return moves in an order based on policy
        if self.move_ordering == "neutral":
            return list(moves)
        ordered = list(moves)
        # move tt_best to front if requested
        if self.move_ordering == "tt_best" and tt_best_move is not None and tt_best_move in ordered:
            ordered.remove(tt_best_move)
            ordered.insert(0, tt_best_move)
            return ordered
        # best-first: heuristic descending
        if self.move_ordering == "best-first" or self.move_ordering == "heuristic":
            scored = [(self._score_move(state, m), m) for m in ordered]
            scored.sort(key=lambda x: x[0], reverse=maximizing)
            return [m for _, m in scored]
        # worst-first: heuristic ascending
        if self.move_ordering == "worst-first":
            scored = [(self._score_move(state, m), m) for m in ordered]
            scored.sort(key=lambda x: x[0], reverse=not maximizing)
            return [m for _, m in scored]
        # random ordering
        if self.move_ordering == "random":
            import random
            random.shuffle(ordered)
            return ordered
        # history heuristic ordering
        if self.move_ordering == "history":
            scored = [(self.history_table.get(m, 0), m) for m in ordered]
            scored.sort(key=lambda x: x[0], reverse=maximizing)
            return [m for _, m in scored]
        # fallback
        return ordered
    # -------------------------------------
    # Alpheta implementation --> Replace self.choose_move_minimax(state, depth) with self.choose_move_alphabeta(state, depth)

    # Alpha beta pruning =
    def alphabeta_value(self, state, depth, alpha=-inf, beta=+inf):
        # check timer for iterative-deepening cutoff
        if getattr(self, 'stop_time', None) is not None and time.perf_counter() > self.stop_time:
            raise TimeoutError()
        self.nodes += 1
        if hasattr(state, "terminal_value"):
            tv = state.terminal_value()
            if tv is not None:
                return tv
        if depth == 0:
            return self.heuristic_evaluation(state)

        key = state.tt_key()
        entry = self.transposition_table.get(key) if self.use_transposition_table else None
        tt_best = None
        if entry:
            stored_depth, flag, value, best_move = entry
            if stored_depth >= depth:
                if flag == "EXACT":
                    return value
                if flag == "LOWER" and value >= beta:
                    return value
                if flag == "UPPER" and value <= alpha:
                    return value
            tt_best = best_move

        moves = state.legal_moves()
        if not moves:
            return 0.0
        moves = self._order_moves(state, moves, tt_best, maximizing=(state.turn_to_move == 0))

        alpha0 = alpha
        if state.turn_to_move == 0:
            best_value = -inf
            best_move = moves[0]
            for move in moves:
                undo = state.make(move)
                val = self.alphabeta_value(state, depth - 1, alpha, beta)
                state.unmake(undo)
                if val > best_value:
                    best_value = val
                    best_move = move
                if best_value > alpha:
                    alpha = best_value
                if alpha >= beta:
                    break
            flag = "EXACT"
            if best_value <= alpha0:
                flag = "UPPER"
            elif best_value >= beta:
                flag = "LOWER"
            self.transposition_table[key] = (depth, flag, best_value, best_move)
            return best_value
        else:
            best_value = +inf
            best_move = moves[0]
            for move in moves:
                undo = state.make(move)
                val = self.alphabeta_value(state, depth - 1, alpha, beta)
                state.unmake(undo)
                if val < best_value:
                    best_value = val
                    best_move = move
                if best_value < beta:
                    beta = best_value
                if alpha >= beta:
                    break
            flag = "EXACT"
            if best_value <= alpha:
                flag = "UPPER"
            elif best_value >= beta:
                flag = "LOWER"
            self.transposition_table[key] = (depth, flag, best_value, best_move)
            return best_value

    def choose_move_alphabeta(self, state, depth):
        self.nodes = 0
        moves = state.legal_moves()
        if not moves:
            return None, 0.0, 0
        if state.turn_to_move == 0:
            best_value = -inf
            best_move = moves[0]
            for move in moves:
                undo = state.make(move)
                val = self.alphabeta_value(state, depth - 1, -inf, +inf)
                state.unmake(undo)
                if val > best_value:
                    best_value = val
                    best_move = move
            return best_move, best_value, self.nodes
        else:
            best_value = +inf
            best_move = moves[0]
            for move in moves:
                undo = state.make(move)
                val = self.alphabeta_value(state, depth - 1, -inf, +inf)
                state.unmake(undo)
                if val < best_value:
                    best_value = val
                    best_move = move
            return best_move, best_value, self.nodes

    def sweep_depths_alphabeta(self, state, max_depth=6, per_depth_budget_sec=25.0):
        results = []
        for depth in range(1, max_depth + 1):
            t0 = time.perf_counter()
            mv, val, nodes = self.choose_move_alphabeta(state, depth)   # !! change mv variable name 
            elapsed = time.perf_counter() - t0
            r = nodes / elapsed if elapsed > 0 else float("nan")   # !! change r variable name 
            pv = state.encode_moves(mv) if mv is not None else None   # !! change pv variable name
            results.append({'depth': depth, 'value': val, 'nodes': nodes, 'elapsed': elapsed, 'nodes_per_sec': r, 'pv_move': pv})
            if elapsed > per_depth_budget_sec:
                break
        return results

    # --- Iterative deepening with time control ---
    def _start_timer(self, time_limit_sec, safety_margin=0.5):
        # sets a stop_time attribute that search value functions check
        self.stop_time = time.perf_counter() + max(0.0, time_limit_sec - safety_margin)

    def _clear_timer(self):
        self.stop_time = None

    def choose_move_minimax_timed(self, state, max_depth=6, time_limit_sec=10.0, safety_margin=0.5):
        """Iterative deepening minimax: return best move found within time limit.

        Returns (best_move, best_value, nodes) where nodes are from the last completed depth.
        """
        best_result = (None, 0.0, 0)
        # iterative deepen
        for depth in range(1, max_depth + 1):
            try:
                self._start_timer(time_limit_sec, safety_margin=safety_margin)
                mv, val, nodes = self.choose_move_minimax(state, depth)
                # completed depth successfully: record result
                best_result = (mv, val, nodes)
            except TimeoutError:
                # time expired during this depth; return last successful result
                break
            finally:
                self._clear_timer()
        return best_result

    def choose_move_alphabeta_timed(self, state, max_depth=6, time_limit_sec=10.0, safety_margin=0.5):
        """Iterative deepening alpha-beta with time control."""
        best_result = (None, 0.0, 0)
        for depth in range(1, max_depth + 1):
            try:
                self._start_timer(time_limit_sec, safety_margin=safety_margin)
                mv, val, nodes = self.choose_move_alphabeta(state, depth)
                best_result = (mv, val, nodes)
            except TimeoutError:
                break
            finally:
                self._clear_timer()
        return best_result

    # --- helpers for threefold-repetition / real-game occurrence counting ---
    def record_position_occurrence(self, key):
        """Increment and return the occurrence count for a real-game position key.

        This is intended to be used only for actual game moves (not for search make/unmake).
        """
        occurrence_count = self.transposition_table_counts.get(key, 0) + 1
        self.transposition_table_counts[key] = occurrence_count
        return occurrence_count

    def clear_position_occurrences(self):
        self.transposition_table_counts.clear()

# The mold between State, Search and Protocol classes 
# You want to go from state, to action, to transitions (updating the GameState). You want to be able to use the GameState interface, in Search to build iterate through game trees, until you come up decision on which move to do.
# ensures that the protocol component relays I / O to state that our client and actions to opponent. 
# learns color from handshake and prescribes it to who you are --> and will create a board state with who's turn_to_move = 1 or 0

class Agent:
    def __init__(self, host, port, game_id, colour, depth, state, search, time_limit_per_move=10.0, safety_margin=0.5):
        self.host = host
        self.port = port
        self.game_id = game_id
        self.colour = colour.lower()
        # keep the original depth parameter for backwards-compatibility but it is ignored for live play
        self.depth = depth
        self.state = state
        self.search = search
        self.side = 0 if self.colour == "white" else 1
        self.protocol = Protocol(self.host, self.port)
        # time control for live play (time-only iterative deepening)
        self.time_limit_per_move = float(time_limit_per_move)
        self.safety_margin = float(safety_margin)
        # initialize real-game occurrence counts (threefold repetition detection)
        try:
            # search is expected to be an Engine instance
            self.search.clear_position_occurrences()
        except Exception:
            pass

    def run(self):
        self.protocol.connect()
        self.protocol.send_line(f"{self.game_id} {self.colour}")
        # record the starting position
        try:
            position_key = self.state.tt_key()
            occurrence_count = self.search.record_position_occurrence(position_key)
            # if starting position already 3 (unlikely), stop immediately
            if occurrence_count >= 3:
                self.protocol.close()
                return
        except Exception:
            pass
        if self.side == 0:
            self._play_our_turn()
        while True:
            try:
                line = self.protocol.recv_line(120.0)
            except Exception:
                break
            if not line:
                break
            try:
                move = self.state.decode_moves(line)
            except Exception:
                continue
            self.state.make(move)
            # record opponent's move as a real-game position
            try:
                position_key = self.state.tt_key()
                occurrence_count = self.search.record_position_occurrence(position_key)
                if occurrence_count >= 3:
                    # threefold repetition reached -> draw
                    break
            except Exception:
                pass
            tv = None
            if hasattr(self.state, "terminal_value"):
                tv = self.state.terminal_value()
            if tv is not None:
                break
            self._play_our_turn()
        self.protocol.close()

    def _play_our_turn(self):
        # use iterative deepening with a 10s per-move limit (0.5s safety margin)
        try:
            # Always ignore Part I static depth. Use time-only iterative deepening alpha-beta.
            # We pass a very large max_depth so that the only limiter is the time budget.
            best_move, best_value, node_count = self.search.choose_move_alphabeta_timed(
                self.state, max_depth=9999, time_limit_sec=self.time_limit_per_move, safety_margin=self.safety_margin)
        except Exception:
            # fallback to single-depth call if timed method not available
            try:
                best_move, best_value, node_count = self.search.choose_move_alphabeta(self.state, 1)
            except Exception:
                best_move, best_value, node_count = None, 0.0, 0
        if best_move is None:
            return
        move_text = self.state.encode_moves(best_move)
        self.state.make(best_move)
        # record our move as a real-game position
        try:
            position_key = self.state.tt_key()
            occurrence_count = self.search.record_position_occurrence(position_key)
            if occurrence_count >= 3:
                # threefold repetition reached; send move then stop
                self.protocol.send_line(move_text)
                return
        except Exception:
            pass
        self.protocol.send_line(move_text)
        try:
            _ = self.protocol.recv_line(10.0)
        except Exception:
            pass

class Protocol:
    
    # For I/O handling, socket connections. 

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.reader = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
        self.reader = self.sock.makefile("r", encoding="utf-8", newline="\n")

    def send_line(self, line):
        data = (line + "\n").encode("utf-8")
        self.sock.sendall(data)

    def recv_line(self, timeout=None):
        if timeout is not None:
            self.sock.settimeout(timeout)
        line = self.reader.readline()
        if line == "":
            raise ConnectionError("server closed")
        return line.rstrip("\n")

    def close(self):
        try:
            if self.reader:
                self.reader.close()
        finally:
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock = None


def bench_opening(depth_max=6):
    gs = GameState(5, 4, 0, 0, 0)
    engine = Engine()
    print("which,depth,value,nodes,elapsed,nodes_per_sec,pv_move")
    for row in engine.sweep_depths(gs, max_depth=depth_max, per_depth_budget_sec=25.0):
        print(f"minimax,{row['depth']},{row['value']},{row['nodes']},{row['elapsed']},{row['nodes_per_sec']},{row['pv_move']}")
    engine.transposition_table.clear()
    for row in engine.sweep_depths_alphabeta(gs, max_depth=depth_max, per_depth_budget_sec=25.0):
        print(f"alphabeta,{row['depth']},{row['value']},{row['nodes']},{row['elapsed']},{row['nodes_per_sec']},{row['pv_move']}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Dynamic Connect-3 agent")
    ap.add_argument("--host", default="156trlinux-1.ece.mcgill.ca")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--game", required=True, help="game ID, e.g., game42")
    ap.add_argument("--colour", required=True, choices=["white", "black"])
    ap.add_argument("--depth", type=int, default=99, help="max depth cap for iterative deepening")
    ap.add_argument("--time", type=float, default=10.0, help="seconds per move")
    ap.add_argument("--safety", type=float, default=0.5, help="reserve this many seconds")
    ap.add_argument("--bench", action="store_true", help="run benchmark instead of playing")
    args = ap.parse_args()

    if args.bench:
        bench_opening(depth_max=args.depth)
    else:
        state = GameState(5, 4, 0, 0, 0)
        engine = Engine()
        # Part I fair defaults (you can flip these later):
        engine.use_transposition_table = False
        engine.move_ordering = "neutral"
        agent = Agent(args.host, args.port, args.game, args.colour,
                      args.depth, state, engine,
                      time_limit_per_move=args.time, safety_margin=args.safety)
        agent.run()