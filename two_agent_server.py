#!/usr/bin/env python3

import argparse
import socket
import selectors
import types
import threading
import sys


class GameRoom:
    def __init__(self, game_id):
        self.game_id = game_id
        self.white = None
        self.black = None

    def ready(self):
        return (self.white is not None) and (self.black is not None)

    def other(self, sock):
        if sock is self.white:
            return self.black
        if sock is self.black:
            return self.white
        return None


def accept_wrapper(sock, sel, rooms):
    conn, addr = sock.accept()
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"", game=None, colour=None)
    sel.register(conn, selectors.EVENT_READ, data=data)
    print(f"[SERVER] accepted connection from {addr}")


def service_connection(key, mask, sel, rooms):
    sock = key.fileobj
    data = key.data
    try:
        if mask & selectors.EVENT_READ:
            recv = sock.recv(4096)
            if recv:
                data.inb += recv
                while b"\n" in data.inb:
                    line, data.inb = data.inb.split(b"\n", 1)
                    text = line.decode("utf-8", errors="ignore").strip()
                    if data.game is None:
                        # expect handshake: "<game_id> <colour>"
                        parts = text.split()
                        if len(parts) >= 2:
                            gid = parts[0]
                            colour = parts[1].lower()
                            data.game = gid
                            data.colour = colour
                            room = rooms.setdefault(gid, GameRoom(gid))
                            if colour == "white":
                                room.white = sock
                            elif colour == "black":
                                room.black = sock
                            else:
                                print(f"[SERVER] unknown colour '{colour}' from {data.addr}")
                                sel.unregister(sock)
                                sock.close()
                                return
                            print(f"[SERVER] registered {colour} for game {gid} from {data.addr}")
                            if room.ready():
                                # notify players that the game is ready (optional)
                                try:
                                    room.white.sendall(b"READY\n")
                                    room.black.sendall(b"READY\n")
                                except Exception:
                                    pass
                        else:
                            print(f"[SERVER] invalid handshake '{text}' from {data.addr}")
                    else:
                        # forward to opponent if available
                        room = rooms.get(data.game)
                        if room is None:
                            continue
                        peer = room.other(sock)
                        if peer is not None:
                            try:
                                peer.sendall(line + b"\n")
                            except Exception:
                                pass
            else:
                # connection closed
                print(f"[SERVER] closing connection to {data.addr}")
                # cleanup rooms
                if data.game and data.game in rooms:
                    room = rooms[data.game]
                    if room.white is sock:
                        room.white = None
                    if room.black is sock:
                        room.black = None
                    if room.white is None and room.black is None:
                        rooms.pop(data.game, None)
                sel.unregister(sock)
                sock.close()
    except ConnectionResetError:
        try:
            sel.unregister(sock)
        except Exception:
            pass
        sock.close()


def run_server(host, port):
    sel = selectors.DefaultSelector()
    rooms = {}
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((host, port))
    lsock.listen()
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)
    print(f"[SERVER] listening on {host}:{port}")
    try:
        while True:
            events = sel.select(timeout=1)
            for key, mask in events:
                if key.data is None:
                    accept_wrapper(key.fileobj, sel, rooms)
                else:
                    service_connection(key, mask, sel, rooms)
    except KeyboardInterrupt:
        print("[SERVER] shutting down")
    finally:
        sel.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=12345)
    args = ap.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple relay server for Dynamic Connect-3 agents.

Usage:
    python three_agent_server.py [--host HOST] [--port PORT]

The server accepts two TCP clients per game id. Clients send a handshake line:
    <game_id> <colour>\n
(e.g. "game99 white\n"). After both clients join the same game id, the server
forwards each newline-terminated message from one client to the other. It also
sends a short "OK" acknowledgement back to the sender so agents waiting for an
ack receive something.

This is intended for local testing only (not production-grade).
"""

import argparse
import socket
import threading
import sys

GAMES = {}   # game_id -> {'white': (conn, fileobj), 'black': (conn, fileobj)}
GAMES_LOCK = threading.Lock()


def handle_client(conn, addr):
    try:
        reader = conn.makefile(mode='r', encoding='utf-8', newline='\n')
        # read handshake
        line = reader.readline()
        if not line:
            conn.close()
            return
        line = line.strip()
        parts = line.split()
        if len(parts) < 2:
            conn.sendall(b"ERR missing-handshake\n")
            conn.close()
            return
        game_id = parts[0]
        colour = parts[1].lower()
        if colour not in ('white', 'black'):
            conn.sendall(b"ERR colour\n")
            conn.close()
            return

        with GAMES_LOCK:
            game = GAMES.setdefault(game_id, {})
            if colour in game:
                conn.sendall(b"ERR duplicate\n")
                conn.close()
                return
            game[colour] = (conn, reader)
            opponent = 'black' if colour == 'white' else 'white'
            print(f"{addr} joined game {game_id} as {colour}")
            if opponent in game:
                print(f"Game {game_id} ready: white and black connected")

        # main loop: forward lines to opponent
        while True:
            line = reader.readline()
            if line == '':
                # client closed
                break
            msg = line.rstrip('\n')
            print(f"[{game_id}] {colour} -> {msg}")
            with GAMES_LOCK:
                game = GAMES.get(game_id, {})
                opp = game.get(opponent)
            if opp:
                opp_conn, _ = opp
                try:
                    opp_conn.sendall((msg + "\n").encode('utf-8'))
                    # ack sender
                    conn.sendall(b"OK\n")
                except Exception:
                    # ignore and continue
                    pass
            else:
                # no opponent yet, just ack so agent can continue
                try:
                    conn.sendall(b"OK\n")
                except Exception:
                    pass

    except Exception as exc:
        print(f"client handler error: {exc}", file=sys.stderr)
    finally:
        # cleanup
        try:
            reader.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        with GAMES_LOCK:
            game = GAMES.get(game_id)
            if game and colour in game:
                del game[colour]
            if game and not game:
                del GAMES[game_id]
        print(f"{addr} disconnected from game {game_id} as {colour}")


def run_server(host='0.0.0.0', port=12345):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(8)
    print(f"two_agent_server listening on {host}:{port}")
    try:
        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        sock.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=12345)
    args = ap.parse_args()
    run_server(host=args.host, port=args.port)
