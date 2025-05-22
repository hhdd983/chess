import chess
import chess.engine
import pygame

pygame.init()
width, height = 480, 480
square_size = width // 8
white = (238, 238, 210)
black = (118, 150, 86)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Chess Game (Joon & Boaz)')

# White pieces
wk = pygame.transform.scale(pygame.image.load(
    'wk.png'), (square_size, square_size))
wq = pygame.transform.scale(pygame.image.load(
    'wq.png'), (square_size, square_size))
wn1 = pygame.transform.scale(pygame.image.load(
    'wn.png'), (square_size, square_size))
wn2 = pygame.transform.scale(pygame.image.load(
    'wn.png'), (square_size, square_size))
wb1 = pygame.transform.scale(pygame.image.load(
    'wb.png'), (square_size, square_size))
wb2 = pygame.transform.scale(pygame.image.load(
    'wb.png'), (square_size, square_size))
wr1 = pygame.transform.scale(pygame.image.load(
    'wr.png'), (square_size, square_size))
wr2 = pygame.transform.scale(pygame.image.load(
    'wr.png'), (square_size, square_size))
wp1 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))
wp2 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))
wp3 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))
wp4 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))
wp5 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))
wp6 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))
wp7 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))
wp8 = pygame.transform.scale(pygame.image.load(
    'wp.png'), (square_size, square_size))

# Black pieces
bk = pygame.transform.scale(pygame.image.load('bk.png'), (square_size, square_size))
bq = pygame.transform.scale(pygame.image.load(
    'bq.png'), (square_size, square_size))
bn1 = pygame.transform.scale(pygame.image.load(
    'bn.png'), (square_size, square_size))
bn2 = pygame.transform.scale(pygame.image.load(
    'bn.png'), (square_size, square_size))
br1 = pygame.transform.scale(pygame.image.load(
    'br.png'), (square_size, square_size))
br2 = pygame.transform.scale(pygame.image.load(
    'br.png'), (square_size, square_size))
bb1 = pygame.transform.scale(pygame.image.load(
    'bb.png'), (square_size, square_size))
bb2 = pygame.transform.scale(pygame.image.load(
    'bb.png'), (square_size, square_size))
bp1 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
bp2 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
bp3 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
bp4 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
bp5 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
bp6 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
bp7 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
bp8 = pygame.transform.scale(pygame.image.load(
    'bp.png'), (square_size, square_size))
board = [[br1, bn1, bb1, bq, bk, bb2, bn2, br2],
         [bp1, bp2, bp3, bp4, bp5, bp6, bp7, bp8],
         [None, None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None, None],
         [None, None, None, None, None, None, None, None],
         [wp1, wp2, wp3, wp4, wp5, wp6, wp7, wp8],
         [wr1, wn1, wb1, wq, wk, wb2, wn2, wr2]]

run = True
selected_square = None
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            from_i,from_j = event.pos[0]//square_size, event.pos[1]//square_size
            if board[from_j][from_i] is not None:
                selected_square = (from_j,from_i)
        elif event.type == pygame.MOUSEBUTTONUP:
            if selected_square:
                to_i, to_j = event.pos[0]//square_size, event.pos[1]//square_size
                board[to_j][to_i] = board[from_j][from_i]
                board[from_j][from_i] = None


    for i in range(8):  # i is row
        for j in range(8):  # j is column=
            color = white if (i + j) % 2 == 0 else black
            x = i * square_size
            y = j * square_size
            pygame.draw.rect(screen, color, (x, y, square_size, square_size))
            piece = board[j][i]
            if piece != None:
                screen.blit(piece, (x, y))

    # White pieces

    pygame.display.flip()

pygame.quit()