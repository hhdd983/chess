# data_collector.py (Lichess API 활용 - 특정 사용자 게임)

import chess
import chess.pgn
import numpy as np
import requests
import time 
import io
import os

def board_to_vector(board):
    board_vector = np.zeros(64 * 12, dtype=np.float32)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_idx = piece_map[piece.symbol()]
            board_vector[i * 12 + piece_idx] = 1
    return board_vector

def move_to_one_hot(move, board):
    from_square = move.from_square
    to_square = move.to_square
    one_hot_move = np.zeros(64 * 64, dtype=np.float32)
    move_idx = from_square * 64 + to_square
    one_hot_move[move_idx] = 1
    return one_hot_move

def fetch_games_for_user(username, max_games=100):
    """
    Lichess API를 사용하여 특정 사용자의 게임을 가져옵니다.
    """
    url = f"https://lichess.org/api/games/user/{username}"
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": "ChessAI-DataCollector/1.0 (contact: your_email@example.com)"
    }
    params = {
        "max": max_games,           # 가져올 게임 수 제한
        "rated": "true",            # 레이팅 게임만
        "perfType": "blitz,rapid",  # 빠른 게임 또는 래피드 게임 (너무 느린 게임은 제외)
        "moves": "true",            # 기보 포함
        "pgnInJson": "false",       # PGN은 JSON 외부에 직접 포함 (기본값)
        "clocks": "false",          # 시계 정보 제외 (데이터 경량화)
        "evals": "false"            # 엔진 평가 정보 제외 (데이터 경량화)
    }

    print(f"Fetching {max_games} games for user: {username}...")
    try:
        response = requests.get(url, headers=headers, params=params, stream=True)
        response.raise_for_status() 

        pgn_data = response.iter_content(chunk_size=1024) 
        
        full_pgn_string = b''.join(pgn_data).decode('utf-8', errors='ignore')
        
        return full_pgn_string

    except requests.exceptions.RequestException as e:
        print(f"Error fetching games for {username}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {username}: {e}")
        return None

def process_pgn_string(pgn_string, username, min_elo_threshold=2200, max_samples_per_user=None):

    X_user_data = []
    y_user_data = []
    collected_samples_this_user = 0

    if not pgn_string:
        return np.array([]), np.array([])

    pgn_io = io.StringIO(pgn_string)
    
    game_count = 0
    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            
            game_count += 1

            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")

            try:
                white_elo = int(white_elo) if white_elo else 0
                black_elo = int(black_elo) if black_elo else 0
            except ValueError:
                continue
            
            board = game.board()
            for move_num, move in enumerate(game.mainline_moves()):
                try:
                    if not board.is_game_over():
                        X_user_data.append(board_to_vector(board))
                        y_user_data.append(move_to_one_hot(move, board))
                        collected_samples_this_user += 1

                        if max_samples_per_user and collected_samples_this_user >= max_samples_per_user:
                            raise StopIteration
                    board.push(move)
                except Exception as e:
                    break
        except StopIteration:
            break
        except Exception as e:
            print(f"  Warning: Skipping game {game_count} for {username} due to PGN parsing error: {e}")
            continue

    print(f"  Processed {game_count} games for {username}. Collected {collected_samples_this_user} samples.")
    return np.array(X_user_data), np.array(y_user_data)


if __name__ == "__main__":
    LICHESS_MASTER_USERNAMES = [
        "DrNykterstein",  # Magnus Carlsen
        "GMHikaru",       # Hikaru Nakamura
        "FabianoCaruana", # Fabiano Caruana
        "MVL",            # Maxime Vachier-Lagrave
    ]

    GAMES_PER_USER = 500

    MAX_SAMPLES_PER_USER = 20000 

    X_total = []
    y_total = []

    for username in LICHESS_MASTER_USERNAMES:
        pgn_string = fetch_games_for_user(username, max_games=GAMES_PER_USER)
        if pgn_string:
            X_user, y_user = process_pgn_string(pgn_string, username, max_samples_per_user=MAX_SAMPLES_PER_USER)
            X_total.extend(X_user)
            y_total.extend(y_user)
        time.sleep(1)

    if X_total:
        X_total = np.array(X_total)
        y_total = np.array(y_total)
        
        np.save('X_train_lichess_api.npy', X_total)
        np.save('y_train_lichess_api.npy', y_total)
        print(f"Data collected via Lichess API saved to X_train_lichess_api.npy and y_train_lichess_api.npy (total {X_total.shape[0]} samples)")
    else:
        print("No data collected. Please check usernames, API connectivity, or rate limits.")