import pygame
import chess
import numpy as np
import tensorflow as tf

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

def predict_best_move(board, model):
    if model is None:
        print("AI model not loaded, cannot predict move.")
        return None

    board_vec = board_to_vector(board)
    input_vec = np.expand_dims(board_vec, axis=0)
    
    predictions = model.predict(input_vec, verbose=0)[0]

    best_move = None
    max_prob = -1.0

    legal_moves_list = list(board.legal_moves)
    if not legal_moves_list:
        return None

    for move in legal_moves_list:
        from_square = move.from_square
        to_square = move.to_square
        move_idx = from_square * 64 + to_square
        
        if move_idx < len(predictions):
            prob = predictions[move_idx]
            if prob > max_prob:
                max_prob = prob
                best_move = move
        else:
            pass
    
    if best_move is None and legal_moves_list:
        print("AI could not find a confident best move, picking first legal move.")
        return legal_moves_list[0]

    return best_move

# Setting
pygame.init()
width, height = 560, 560
square_size = width // 8
white = (238, 238, 210)
black = (118, 150, 86)
text_color = (200, 0, 0)

font = pygame.font.SysFont(None, 48)

screen = pygame.display.set_mode((width+280, height))
pygame.display.set_caption('Chess Game (Joon & Boaz)')
clock = pygame.time.Clock()

piece_images = {}

def parser(s):
    if s[0] == 'w':
        return chr(ord(s[1])-32)
    else:
        return s[1]

for symbol in ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']:
    try:
        img = pygame.image.load(f'{symbol}.png')
        piece_images[parser(symbol)] = pygame.transform.scale(
            img, (square_size, square_size))
    except pygame.error as e:
        print(f"Error loading image {symbol}.png: {e}")
        print("Please ensure all piece images (e.g., bp.png, wk.png) are in the same directory as the script.")
        pygame.quit()
        exit()


board = chess.Board()
files = 'abcdefgh'

def coord_to_uci(col, row):
    return files[col] + str(8 - row)

if __name__ == "__main__":
    chess_ai_model = None
    try:
        chess_ai_model = tf.keras.models.load_model('chess_ai_model.h5')
        print("Chess AI model loaded successfully.")
    except Exception as e:
        print(f"Error loading AI model: {e}")
        print("Please run model_trainer.py first to train and save the model as 'chess_ai_model.h5'.")

    selected_sq = None
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                col, row = event.pos[0] // square_size, event.pos[1] // square_size
                uci_sq = coord_to_uci(col, row)
                idx = chess.square(col, 7 - row)
                piece = board.piece_at(idx)
                if piece and piece.color == board.turn:
                    selected_sq = uci_sq

            elif event.type == pygame.MOUSEBUTTONUP and selected_sq:
                col, row = event.pos[0] // square_size, event.pos[1] // square_size
                dest_sq = coord_to_uci(col, row)
                move = None
                if dest_sq != selected_sq:
                    try:
                        move = chess.Move.from_uci(selected_sq + dest_sq)
                    except ValueError:
                        move = None

                if move and move in board.legal_moves:
                    board.push(move)
                    selected_sq = None
                    
                    if not board.is_game_over() and board.turn == chess.BLACK:
                        print("AI thinking...")
                        ai_move = predict_best_move(board, chess_ai_model)
                        if ai_move:
                            print(f"AI plays: {ai_move.uci()}")
                            board.push(ai_move)
                        else:
                            print("AI could not find a legal move.")
                            if not list(board.legal_moves):
                                print("No legal moves for AI.")
                    elif chess_ai_model is None:
                        print("AI model not loaded. Skipping AI turn.")


        # Print chess board
        for r in range(8):
            for c in range(8):
                color = white if (r + c) % 2 == 0 else black
                rect = (c * square_size, r * square_size, square_size, square_size)
                pygame.draw.rect(screen, color, rect)
        rect = (560, 0, 280, 560)
        pygame.draw.rect(screen, black, rect)

        # Pieces
        for sq_idx, piece in board.piece_map().items():
            file = chess.square_file(sq_idx)
            rank = chess.square_rank(sq_idx)
            col = file
            row = 7 - rank
            img = piece_images[piece.symbol()]
            screen.blit(img, (col * square_size, row * square_size))

        if board.is_game_over():
            result = board.result()
            if result == '1-0':
                msg = "White wins!"
            elif result == '0-1':
                msg = "Black wins!"
            else:
                msg = "Draw!"

            text_surf = font.render(msg, True, text_color)
            text_rect = text_surf.get_rect(center=(width + 280 // 2, height // 2))
            overlay = pygame.Surface(
                (text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 180))
            screen.blit(overlay, (text_rect.x - 10, text_rect.y - 10))
            screen.blit(text_surf, text_rect)
        pygame.display.flip()

    pygame.quit()