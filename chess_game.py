import pygame
import chess
import numpy as np
import tensorflow as tf


# ---------- NN helpers ----------
def board_to_vector(board: chess.Board) -> np.ndarray:
    """One-hot encode the 8×8×12 board for the network."""
    board_vector = np.zeros(64 * 12, dtype=np.float32)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            idx = piece_map[piece.symbol()]
            board_vector[sq * 12 + idx] = 1
    return board_vector


def predict_best_move(board: chess.Board, model) -> chess.Move | None:
    # Return the legal move with the highest probability.
    if model is None:
        print("AI model not loaded, cannot predict move.")
        return None

    board_vec = board_to_vector(board).reshape((8, 8, 12))
    board_input = np.expand_dims(board_vec, axis=0)
    preds = model.predict(board_input, verbose=0)[0]

    best_move, best_prob = None, -1.0
    for mv in board.legal_moves:
        move_idx = mv.from_square * 64 + mv.to_square
        if move_idx < len(preds) and preds[move_idx] > best_prob:
            best_prob, best_move = preds[move_idx], mv

    if best_move is None:
        print("AI couldn’t decide; choosing first legal move.")
        return next(iter(board.legal_moves), None)
    return best_move


# ---------- Pygame setup ----------
pygame.init()
WIDTH, HEIGHT = 560, 560           # 8 × 70 px squares
SQUARE = WIDTH // 8
LIGHT = (238, 238, 210)
DARK = (118, 150,  86)
TEXT = (200,   0, 255)

screen = pygame.display.set_mode((WIDTH + 280, HEIGHT))
pygame.display.set_caption('Chess Game (Joon & Boaz)')
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# ---------- button ----------


def make_button_surface(ai_white: bool) -> pygame.Surface:
    label = "Play as Black" if ai_white else "Play as White"
    return font.render(label, True, (0, 0, 0), TEXT)


button_surf = make_button_surface(False)
text_rect = button_surf.get_rect()
padding = 10

center_x, center_y = 700, 240
button_width = text_rect.width + padding * 2
button_height = text_rect.height + padding * 2
button_x = center_x - button_width // 2
button_y = center_y - button_height // 2


# ---------- load images ----------


def parser(s: str) -> str:
    return chr(ord(s[1])-32) if s[0] == 'w' else s[1]


piece_images, piece_images_outline = {}, {}
for sym in [
    'wp', 'wn', 'wb', 'wr', 'wq', 'wk',
    'bp', 'bn', 'bb', 'br', 'bq', 'bk'
]:
    img = pygame.image.load(f'images/{sym}.png')
    piece_images[parser(sym)] = pygame.transform.scale(img, (SQUARE, SQUARE))
    img = pygame.image.load(f'images/{sym}_outlined.png')
    piece_images_outline[parser(sym)] = pygame.transform.scale(
        img, (SQUARE, SQUARE))

# ---------- chess board ----------
board = chess.Board()
FILES = 'abcdefgh'


def coord_to_uci(col: int, row: int) -> str:
    return FILES[col] + str(8 - row)


# ---------- load / warn about model ----------
model = None
try:
    model = tf.keras.models.load_model('chess_ai_model.keras')
    print("Chess AI model loaded.")
except Exception as e:
    print(f"[WARN] Couldn’t load model: {e}\nRun model_trainer.py first.")

# ---------- game loop ----------
ai_plays_white = False         # AI plays Black by default
selected_sq = None
running = True

while running:
    clock.tick(60)

    # AI makes a move whenever it’s its turn
    if not board.is_game_over():
        if (ai_plays_white and board.turn == chess.WHITE) or \
           (not ai_plays_white and board.turn == chess.BLACK):
            ai_move = predict_best_move(board, model)
            if ai_move:
                board.push(ai_move)

    # handle user events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # --- button click toggles sides ---
            if button_rect.collidepoint(event.pos):
                ai_plays_white = not ai_plays_white
                selected_sq = None
                button_surf = make_button_surface(ai_plays_white)
                continue  # skip board-click handling

            # ignore clicks outside board
            if event.pos[0] >= WIDTH or event.pos[1] >= HEIGHT:
                continue

            col, row = event.pos[0] // SQUARE, event.pos[1] // SQUARE
            idx = chess.square(col, 7 - row)
            piece = board.piece_at(idx)
            if piece and piece.color == board.turn:
                selected_sq = coord_to_uci(col, row)

        elif event.type == pygame.MOUSEBUTTONUP and selected_sq:
            # ignore releases outside board
            if event.pos[0] >= WIDTH or event.pos[1] >= HEIGHT:
                selected_sq = None
                continue

            col, row = event.pos[0] // SQUARE, event.pos[1] // SQUARE
            dest = coord_to_uci(col, row)
            if dest != selected_sq:
                try:
                    mv = chess.Move.from_uci(selected_sq + dest)
                    if mv in board.legal_moves:
                        board.push(mv)
                except ValueError:
                    pass
            selected_sq = None

    # draw everything --------------------------------------------------
    #  board squares
    for r in range(8):
        for c in range(8):
            pygame.draw.rect(
                screen,
                LIGHT if (r + c) % 2 == 0 else DARK,
                (c * SQUARE, r * SQUARE, SQUARE, SQUARE)
            )

    #  right-hand panel
    pygame.draw.rect(screen, DARK, (WIDTH, 0, 280, HEIGHT))
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

    # Draw button
    pygame.draw.rect(screen, TEXT, button_rect, border_radius=8)
    screen.blit(button_surf, (button_x + padding, button_y + padding))

    #  pieces
    for sq, piece in board.piece_map().items():
        col, row = chess.square_file(sq), 7 - chess.square_rank(sq)
        img = piece_images_outline[piece.symbol()] if \
            selected_sq == coord_to_uci(col, row) else \
            piece_images[piece.symbol()]
        screen.blit(img, (col * SQUARE, row * SQUARE))

    #  game-over message
    if board.is_game_over():
        result = board.result()
        msg = "White wins!" if result == '1-0' else \
              "Black wins!" if result == '0-1' else "Draw!"
        txt_surf = font.render(msg, True, TEXT)
        txt_rect = txt_surf.get_rect(center=(WIDTH + 140, HEIGHT // 2))
        overlay = pygame.Surface(
            (txt_rect.w + 20, txt_rect.h + 20), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 180))
        screen.blit(overlay, (txt_rect.x - 10, txt_rect.y - 10))
        screen.blit(txt_surf, txt_rect)

    pygame.display.flip()

pygame.quit()
