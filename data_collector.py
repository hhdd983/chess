"""
pgnmentor_dataset_converter.py

Convert all PGN files from a local PGN Mentor download into NumPy datasets.
Each move becomes a sample: X is a 768-dim board vector (8×8×12 one-hot),
Y is a 4096-dim one-hot move vector (from_square*64 + to_square).

Usage:
    python pgnmentor_dataset_converter.py

Requires:
    pip install numpy chess

Outputs:
    X_players.npy  # shape (N, 768)
    y_players.npy  # shape (N, 4096)
"""
import os
import chess
import chess.pgn
import numpy as np


def board_to_vector(board: chess.Board) -> np.ndarray:
    vec = np.zeros(64 * 12, dtype=np.int8)
    mapping = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            idx = mapping[piece.symbol()]
            vec[sq * 12 + idx] = 1
    return vec


def move_to_one_hot(move: chess.Move) -> np.ndarray:
    """One-hot encode move into shape (4096,) vector."""
    one_hot = np.zeros(64 * 64, dtype=np.int8)
    idx = move.from_square * 64 + move.to_square
    one_hot[idx] = 1
    return one_hot


def process_pgn_file(path: str, max_samples: int = None):
    """
    Parse a single PGN file, converting each legal move into X,y samples.
    Returns:
        X: np.ndarray of shape (M, 768)
        y: np.ndarray of shape (M, 4096)
    """
    X_list, y_list = [], []
    samples = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for mv in game.mainline_moves():
                if max_samples and samples >= max_samples:
                    return np.array(X_list), np.array(y_list)
                X_list.append(board_to_vector(board))
                y_list.append(move_to_one_hot(mv))
                board.push(mv)
                samples += 1
    return np.array(X_list), np.array(y_list)


def main():
    PGN_DIR = 'pgnmentor_tournaments'  # change if your directory differs
    if not os.path.isdir(PGN_DIR):
        raise SystemExit(
            f"Directory '{PGN_DIR}' not found. Place your .pgn files there.")

    X_all, y_all = [], []
    print(f"Scanning directory: {PGN_DIR}")
    for fname in os.listdir(PGN_DIR):
        if not fname.lower().endswith('.pgn'):
            continue
        path = os.path.join(PGN_DIR, fname)
        print(f"Processing {fname}...")
        X, y = process_pgn_file(path)
        if X.size:
            X_all.append(X)
            y_all.append(y)
        print(f" → {X.shape[0]} samples from {fname}")

    if not X_all:
        raise SystemExit("No PGN data found or no samples generated.")

    X_final = np.vstack(X_all)
    y_final = np.vstack(y_all)
    print(f"Total samples: {X_final.shape[0]}")

    np.save('X_players.npy', X_final)
    np.save('y_players.npy', y_final)
    print("Saved X_players.npy and y_players.npy")


if __name__ == '__main__':
    main()
