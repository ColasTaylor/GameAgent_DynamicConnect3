

import socket, sys, time, json
from math import inf
from itertools import combinations, permutations

# ---------------- Game state ----------------
class GameState:
    def __init__(self, width, height, white_init=0, black_init=0, turn=0):
        self.board_width, self.board_height = width, height
        self.white_bitboard, self.black_bitboard = white_init, black_init
        self.turn_to_move = turn
        self.board_mask = (1 << (width * height)) - 1
        self.neighbours = self._precompute_neighbours()
        self.terminal_states = self._precompute_terminal_states()
        if white_init == 0 and black_init == 0:
            self._init_start()

    def _init_start(self):
        white_coords = [(1,1),(5,2),(1,3),(5,4)]
        black_coords = [(5,1),(1,2),(5,3),(1,4)]
        if (self.board_width, self.board_height) == (7,6):
            white_coords = [(x+1,y+1) for x,y in white_coords]; black_coords = [(x+1,y+1) for x,y in black_coords]
        elif (self.board_width, self.board_height) != (5,4):
            raise ValueError("Unsupported board size")
        white_bits = black_bits = 0
        for x,y in white_coords: white_bits |= 1 << self.to_index(x,y)
        for x,y in black_coords: black_bits |= 1 << self.to_index(x,y)
        self.white_bitboard = white_bits & self.board_mask
        self.black_bitboard = black_bits & self.board_mask
        self.turn_to_move = 0

    def inbounds(self, x, y): 
        return 1 <= x <= self.board_width and 1 <= y <= self.board_height

    def to_index(self, x, y): 
        return (y-1)*self.board_width + (x-1)

    def _precompute_neighbours(self):
        dirs = {"N":(0,-1),"S":(0,1),"E":(1,0),"W":(-1,0)}
        nb = [None]*(self.board_width*self.board_height)
        for y in range(1,self.board_height+1):
            for x in range(1,self.board_width+1):
                i = self.to_index(x,y)
                d = {}
                for k,(dx,dy) in dirs.items():
                    nx,ny = x+dx,y+dy
                    if self.inbounds(nx,ny):
                        d[k] = self.to_index(nx,ny)
                nb[i]=d
        return nb

    def _precompute_terminal_states(self):
        masks=[]; dirs=[(1,0),(0,1),(1,1),(1,-1)]
        for y in range(1,self.board_height+1):
            for x in range(1,self.board_width+1):
                for dx,dy in dirs:
                    valid=True; mask=0
                    for step in range(3):
                        nx,ny = x+step*dx, y+step*dy
                        if not self.inbounds(nx,ny): valid=False; break
                        mask |= 1 << self.to_index(nx,ny)
                    if valid: masks.append(mask)
        return masks

    def terminal_value(self):
        INF=float("inf")
        for m in self.terminal_states:
            if (self.white_bitboard & m)==m: return +INF
            if (self.black_bitboard & m)==m: return -INF
        return None

    def legal_moves(self):
        moves=[]; bits=self.white_bitboard if self.turn_to_move==0 else self.black_bitboard
        occ=(self.white_bitboard|self.black_bitboard)&self.board_mask
        N=self.board_width*self.board_height
        for i in range(N):
            if ((bits>>i)&1)==0: continue
            for dst in self.neighbours[i].values():
                if ((occ>>dst)&1)==0: moves.append((i,dst))
        return moves

    def make(self, move):
        s,d=move
        if d not in self.neighbours[s].values(): raise ValueError("not neighbour")
        occ=(self.white_bitboard|self.black_bitboard)&self.board_mask
        if ((occ>>d)&1)!=0: raise ValueError("occupied")
        if self.turn_to_move==0:
            if ((self.white_bitboard>>s)&1)==0: raise ValueError("no white")
            self.white_bitboard ^= (1<<s)|(1<<d)
        else:
            if ((self.black_bitboard>>s)&1)==0: raise ValueError("no black")
            self.black_bitboard ^= (1<<s)|(1<<d)
        self.turn_to_move ^= 1
        return (s,d)

    def unmake(self, undo):
        s,d=undo; self.turn_to_move^=1
        if self.turn_to_move==0: self.white_bitboard ^= (1<<s)|(1<<d)
        else: self.black_bitboard ^= (1<<s)|(1<<d)

    def decode_moves(self, txt):
        token = txt.strip().upper().replace(" ","")
        if len(token) < 3: raise ValueError("bad move")
        x_coord, y_coord = int(token[0]), int(token[1]); direction = token[2]
        if not self.inbounds(x_coord, y_coord): raise ValueError("oob")
        source_index = self.to_index(x_coord, y_coord)
        if direction not in self.neighbours[source_index]: raise ValueError("dir")
        return (source_index, self.neighbours[source_index][direction])

    def encode_moves(self, move):
        source, destination = move
        x1 = (source % self.board_width) + 1
        y1 = (source // self.board_width) + 1
        x2 = (destination % self.board_width) + 1
        y2 = (destination // self.board_width) + 1
        dx, dy = x2 - x1, y2 - y1
        if (dx, dy) == (1, 0): ch = "E"
        elif (dx, dy) == (-1, 0): ch = "W"
        elif (dx, dy) == (0, 1): ch = "S"
        elif (dx, dy) == (0, -1): ch = "N"
        else: raise ValueError("bad")
        return f"{x1}{y1}{ch}"

    def tt_key(self):
        return (self.white_bitboard,self.black_bitboard,self.turn_to_move,self.board_width,self.board_height)

    def display(self):
        for y in range(1,self.board_height+1):
            row=[]
            for x in range(1,self.board_width+1):
                i=self.to_index(x,y)
                row.append("0" if (self.white_bitboard>>i)&1 else "1" if (self.black_bitboard>>i)&1 else " ")
            print(" , ".join(row))

# ---------------- Engine (search + eval) ----------------
class Engine:
    def __init__(self):
        self.nodes=0
        self.move_ordering="best-first"   # default
        self.eval_mode="improved"
        # default learned weights; override with --weights
        self.theta={
            "two_in_row":1.6855,
            "two_per_line":1.1093,
            "single_cell":0.5494,
            "blocked_lines":0.2316,
            "mobility":0.6488,
            "distance":0.9395,
        }
        self._mask_cache={}
        self.position_counts={}
        self.stop_time=None
        self.algo="ab"  # "ab" or "minimax"

    # ---- evaluation
    def heuristic_evaluation(self, state):
        return self._heuristic_naive(state) if self.eval_mode=="naive" else self._heuristic_improved(state)

    def _line_masks(self, w,h,L):
        key=("line_masks",w,h,L)
        if key in self._mask_cache: return self._mask_cache[key]
        masks=[]; dirs=[(1,0),(0,1),(1,1),(1,-1)]
        def ok(x,y): return 1<=x<=w and 1<=y<=h
        def idx(x,y): return (y-1)*w+(x-1)
        for y in range(1,h+1):
            for x in range(1,w+1):
                for dx,dy in dirs:
                    m=0; good=True
                    for s in range(L):
                        nx,ny=x+s*dx,y+s*dy
                        if not ok(nx,ny): good=False; break
                        m |= 1<<idx(nx,ny)
                    if good: masks.append(m)
        self._mask_cache[key]=masks; return masks

    def _heuristic_naive(self, state):
        masks=self._line_masks(state.board_width,state.board_height,2)
        W,B=state.white_bitboard,state.black_bitboard
        sc=0
        for m in masks:
            if (W&m)==m and (B&m)==0: sc+=1
            elif (B&m)==m and (W&m)==0: sc-=1
        return float(sc)

    def _line_triplets(self,w,h):
        key=("triplets3",w,h)
        if key in self._mask_cache: return self._mask_cache[key]
        trips=[]; dirs=[(1,0),(0,1),(1,1),(1,-1)]
        def ok(x,y): return 1<=x<=w and 1<=y<=h
        def idx(x,y): return (y-1)*w+(x-1)
        for y in range(1,h+1):
            for x in range(1,w+1):
                for dx,dy in dirs:
                    cells=[]; good=True
                    for s in range(3):
                        nx,ny=x+s*dx,y+s*dy
                        if not ok(nx,ny): good=False; break
                        cells.append(idx(nx,ny))
                    if good: trips.append(tuple(cells))
        self._mask_cache[key]=trips; return trips

    def _mobility_for_side(self, state_obj, side):
        side_bits = state_obj.white_bitboard if side == 0 else state_obj.black_bitboard
        occupied = (state_obj.white_bitboard | state_obj.black_bitboard) & state_obj.board_mask
        move_count = 0
        for index in range(state_obj.board_width * state_obj.board_height):
            if ((side_bits >> index) & 1) == 0:
                continue
            for dst in state_obj.neighbours[index].values():
                if ((occupied >> dst) & 1) == 0:
                    move_count += 1
        return move_count

    def _idx_to_xy(self, i,w): return (i%w)+1,(i//w)+1

    def _alignment_distance_uncontested_for_side(self, state_obj, side):
        width, height = state_obj.board_width, state_obj.board_height
        our_bits = state_obj.white_bitboard if side == 0 else state_obj.black_bitboard
        our_indices = [i for i in range(width * height) if ((our_bits >> i) & 1)]
        if len(our_indices) < 3:
            return 3 * ((width - 1) + (height - 1))
        best_cost = 3 * ((width - 1) + (height - 1))
        for triplet_line in self._line_triplets(width, height):
            for chosen in combinations(our_indices, 3):
                for assignment in permutations(chosen, 3):
                    cost = 0
                    for src_idx, dst_idx in zip(assignment, triplet_line):
                        x1, y1 = self._idx_to_xy(src_idx, width)
                        x2, y2 = self._idx_to_xy(dst_idx, width)
                        cost += abs(x1 - x2) + abs(y1 - y2)
                        if cost >= best_cost:
                            break
                    if cost < best_cost:
                        best_cost = cost
        return best_cost

    def _extract_features(self, state_obj):
        white_bits = state_obj.white_bitboard
        black_bits = state_obj.black_bitboard
        width, height = state_obj.board_width, state_obj.board_height
        triplets = self._line_triplets(width, height)
        total_triplets = max(1, len(triplets))
        two_in_row_white = two_in_row_black = two_per_white = two_per_black = single_white = single_black = blocked_lines_count = 0
        for i, j, k in triplets:
            count_white = ((white_bits >> i) & 1) + ((white_bits >> j) & 1) + ((white_bits >> k) & 1)
            count_black = ((black_bits >> i) & 1) + ((black_bits >> j) & 1) + ((black_bits >> k) & 1)
            if count_white > 0 and count_black > 0:
                blocked_lines_count += 1
            if count_white == 1 and count_black == 0:
                single_white += 1
            if count_black == 1 and count_white == 0:
                single_black += 1
            if count_white == 2 and count_black == 0:
                two_per_white += 1
            if count_black == 2 and count_white == 0:
                two_per_black += 1
            if count_black == 0:
                if ((white_bits >> i) & 1) and ((white_bits >> j) & 1) and (((white_bits >> k) & 1) == 0):
                    two_in_row_white += 1
                if ((white_bits >> j) & 1) and ((white_bits >> k) & 1) and (((white_bits >> i) & 1) == 0):
                    two_in_row_white += 1
            if count_white == 0:
                if ((black_bits >> i) & 1) and ((black_bits >> j) & 1) and (((black_bits >> k) & 1) == 0):
                    two_in_row_black += 1
                if ((black_bits >> j) & 1) and ((black_bits >> k) & 1) and (((black_bits >> i) & 1) == 0):
                    two_in_row_black += 1
        two_in_row = (two_in_row_white - two_in_row_black) / total_triplets
        two_per_line = (two_per_white - two_per_black) / total_triplets
        single_cell = (single_white - single_black) / total_triplets
        blocked_fraction = blocked_lines_count / total_triplets
        mobility_diff = (self._mobility_for_side(state_obj, 0) - self._mobility_for_side(state_obj, 1)) / 16.0
        max_total = 3 * ((width - 1) + (height - 1))
        dist_white = self._alignment_distance_uncontested_for_side(state_obj, 0)
        dist_black = self._alignment_distance_uncontested_for_side(state_obj, 1)
        distance_feature = (dist_black - dist_white) / float(max_total)
        return {
            "two_in_row": two_in_row,
            "two_per_line": two_per_line,
            "single_cell": single_cell,
            "blocked_lines": blocked_fraction,
            "mobility": mobility_diff,
            "distance": distance_feature,
        }

    def _heuristic_improved(self, state):
        f=self._extract_features(state); s=0.0
        for k,v in self.theta.items(): s += v*f[k]
        return float(s)

    # ordering
    def _score_move(self, st, mv):
        u=st.make(mv); sc=self.heuristic_evaluation(st); st.unmake(u); return sc

    def _order_moves(self, st, moves, maximizing=True):
        if self.move_ordering=="neutral": return list(moves)
        scored=[(self._score_move(st,m),m) for m in moves]
        scored.sort(key=lambda x:x[0], reverse=maximizing)
        return [m for _,m in scored]

    # search
    def minimax_value(self, st, depth):
        if self.stop_time and time.perf_counter()>self.stop_time: raise TimeoutError()
        self.nodes+=1
        tv=st.terminal_value()
        if tv is not None: return tv
        if depth==0: return self.heuristic_evaluation(st)
        moves=st.legal_moves()
        if not moves: return 0.0
        moves=self._order_moves(st,moves, maximizing=(st.turn_to_move==0))
        if st.turn_to_move==0:
            best=-inf
            for m in moves:
                u=st.make(m)
                try: v=self.minimax_value(st,depth-1)
                finally: st.unmake(u)
                if v>best: best=v
            return best
        else:
            best=+inf
            for m in moves:
                u=st.make(m)
                try: v=self.minimax_value(st,depth-1)
                finally: st.unmake(u)
                if v<best: best=v
            return best

    def alphabeta_value(self, st, depth, alpha=-inf, beta=+inf):
        if self.stop_time and time.perf_counter()>self.stop_time: raise TimeoutError()
        self.nodes+=1
        tv=st.terminal_value()
        if tv is not None: return tv
        if depth==0: return self.heuristic_evaluation(st)
        moves=st.legal_moves()
        if not moves: return 0.0
        moves=self._order_moves(st,moves, maximizing=(st.turn_to_move==0))
        if st.turn_to_move==0:
            best=-inf
            for m in moves:
                u=st.make(m)
                try: v=self.alphabeta_value(st,depth-1,alpha,beta)
                finally: st.unmake(u)
                if v>best: best=v
                if best>alpha: alpha=best
                if alpha>=beta: break
            return best
        else:
            best=+inf
            for m in moves:
                u=st.make(m)
                try: v=self.alphabeta_value(st,depth-1,alpha,beta)
                finally: st.unmake(u)
                if v<best: best=v
                if best<beta: beta=best
                if alpha>=beta: break
            return best

    # iterative deepening with time control
    def _deadline(self, secs, safety=0.5): 
        return time.perf_counter()+max(0.0,secs-safety)

    def choose_move_timed(self, st, secs=10.0, max_depth=9999, safety=0.5):
        self.nodes=0
        best=(None,0.0,0)
        deadline=self._deadline(secs,safety)
        for d in range(1,max_depth+1):
            try:
                self.stop_time=deadline
                s=st.clone()
                if self.algo=="minimax":
                    mv=self._root_minimax(s,d)
                else:
                    mv=self._root_ab(s,d)
                best=mv
                if time.perf_counter()>=deadline: break
            except TimeoutError:
                break
            finally:
                self.stop_time=None
        return best

    def _root_minimax(self, st, depth):
        moves=st.legal_moves()
        if not moves: return (None,0.0,0)
        moves=self._order_moves(st,moves, maximizing=(st.turn_to_move==0))
        if st.turn_to_move==0:
            best=-inf; bestm=moves[0]
            for m in moves:
                u=st.make(m); v=self.minimax_value(st,depth-1); st.unmake(u)
                if v>best: best, bestm=v, m
            return (bestm,best,self.nodes)
        else:
            best=+inf; bestm=moves[0]
            for m in moves:
                u=st.make(m); v=self.minimax_value(st,depth-1); st.unmake(u)
                if v<best: best, bestm=v, m
            return (bestm,best,self.nodes)

    def _root_ab(self, st, depth):
        moves=st.legal_moves()
        if not moves: return (None,0.0,0)
        moves=self._order_moves(st,moves, maximizing=(st.turn_to_move==0))
        if st.turn_to_move==0:
            best=-inf; bestm=moves[0]
            for m in moves:
                u=st.make(m); v=self.alphabeta_value(st,depth-1,-inf,+inf); st.unmake(u)
                if v>best: best, bestm=v, m
            return (bestm,best,self.nodes)
        else:
            best=+inf; bestm=moves[0]
            for m in moves:
                u=st.make(m); v=self.alphabeta_value(st,depth-1,-inf,+inf); st.unmake(u)
                if v<best: best, bestm=v, m
            return (bestm,best,self.nodes)

    # simple repetition counter for draws
    def record_position_occurrence(self, key):
        c=self.position_counts.get(key,0)+1
        self.position_counts[key]=c
        return c
    def clear_position_occurrences(self): self.position_counts.clear()

# Protocol & Agent

class Protocol:
    def __init__(self, host, port): self.host, self.port, self.sock, self.reader = host, port, None, None
    def connect(self):
        self.sock = socket.create_connection((self.host,self.port), timeout=5.0)
        self.reader = self.sock.makefile("r", encoding="utf-8", newline="\n")
    def send_line(self, line): self.sock.sendall((line+"\n").encode("utf-8"))
    def recv_line(self, timeout=None):
        if timeout is not None: self.sock.settimeout(timeout)
        line=self.reader.readline()
        if line=="": raise ConnectionError("server closed")
        return line.rstrip("\n")
    def close(self):
        try:
            if self.reader: self.reader.close()
        finally:
            if self.sock:
                try: self.sock.close()
                finally: self.sock=None

class Agent:
    def __init__(self, host, port, game_id, colour, engine, time_limit=10.0, safety=0.5, verbose=False):
        self.state = GameState(5,4,0,0,0)
        self.engine = engine
        self.time_limit, self.safety, self.verbose = float(time_limit), float(safety), bool(verbose)
        self.colour = colour.lower(); self.side = 0 if self.colour=="white" else 1
        self.proto = Protocol(host, port); self.game_id = game_id
        self._last_sent = None

    def _canon(self, s):
        try: return self.state.encode_moves(self.state.decode_moves(s))
        except: return None

    def run(self):
        self.proto.connect()
        self.proto.send_line(f"{self.game_id} {self.colour}")
        if self.side==0: self._play_our_turn()
        while True:
            try: line = self.proto.recv_line(120.0)
            except Exception: break
            if not line: continue
            # skip server echo
            if self._last_sent and self._canon(line)==self._canon(self._last_sent):
                self._last_sent=None; continue
            if not self._apply_incoming(line): continue
            self._play_our_turn()
        self.proto.close()

    def _apply_incoming(self, line):
        try:
            mv=self.state.decode_moves(line)
            self.state.make(mv)
            if self.verbose:
                print(f"[RECV] {line.strip()}"); self.state.display()
            if self.engine.record_position_occurrence(self.state.tt_key())>=3: return False
            return self.state.terminal_value() is None
        except Exception:
            return False

    def _play_our_turn(self):
        try:
            mv,val,_= self.engine.choose_move_timed(self.state, secs=self.time_limit, max_depth=9999, safety=self.safety)
        except Exception:
            mv=None
        if mv is None: return
        text=self.state.encode_moves(mv)
        try: self.state.make(mv)
        except Exception: return
        if self.verbose:
            print(f"[PLAY] {text}  (v={val:.3f})"); self.state.display()
        if self.engine.record_position_occurrence(self.state.tt_key())>=3:
            self.proto.send_line(text); return
        self.proto.send_line(text); self._last_sent=text

# Main

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(description="Dynamic Connect-3 agent (gameplay)")
    ap.add_argument("--host", default="156trlinux-1.ece.mcgill.ca")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--game", required=False, default="game0")
    ap.add_argument("--colour", choices=["white","black"], default="white")
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--safety", type=float, default=0.5)
    ap.add_argument("--algo", choices=["ab","minimax"], default="ab")
    ap.add_argument("--heuristic", choices=["naive","improved"], default="improved")
    ap.add_argument("--weights", type=str, help="JSON file with learned weights for improved eval")
    ap.add_argument("--verbose", action="store_true")
    args=ap.parse_args()

    eng = Engine()
    eng.algo = args.algo
    eng.eval_mode = args.heuristic
    eng.move_ordering = "best-first"
    if args.heuristic == "improved" and args.weights:
        import json
        with open(args.weights, "r", encoding="utf-8") as f:
            w = json.load(f)
        if hasattr(eng, "set_weights"):
            eng.set_weights(**w)
        else:
            for k, v in w.items():
                if k in eng.theta:
                    eng.theta[k] = float(v)

    agent=Agent(args.host,args.port,args.game,args.colour,eng,time_limit=args.time,safety=args.safety,verbose=args.verbose)
    agent.run()