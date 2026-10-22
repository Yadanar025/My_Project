from operator import iand


PLAYER_WHITE = 'W'
PLAYER_BLACK = 'B'
EMPTY = ' '
KING_WHITE = 'KW'
KING_BLACK = 'KB'

class MMChecker:

    def __init__(self,board=None,current_player=None):
        self.for_king = []
        if board is None:
            self.board = [' ' for _ in range(64)]
            self.set_board()
        else:
            self.board = board
        if current_player is None:
            self.current_player = PLAYER_WHITE
        else:
            self.current_player = current_player

    def create_board(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                board[i][j] = self.board[i * 8 + j]

        return board

    def set_board(self):
        """Sets up the initial state of the board with pieces in their starting positions."""
        for row in range(8):
            for col in range(8):
                if ((row+col)%2==1):
                    self.board[row * 8 + col] = PLAYER_WHITE if (row<=2) else PLAYER_BLACK if (row>=5) else EMPTY
                
    def print_board(self):
        """Prints the current state of the board in a readable format.
        """
        board = self.create_board()
        for row in board:
            print(' '.join(row))


    def get_available_moves(self,current_player):
        """
        Returns a list of available moves on the board.
        """
        available_moves = []
        opponent = PLAYER_BLACK if current_player == PLAYER_WHITE else PLAYER_WHITE
        other_king = KING_BLACK if current_player == PLAYER_WHITE else KING_WHITE
        own_king = KING_WHITE if current_player == PLAYER_WHITE else KING_BLACK

        for i in range(len(self.board)):
            row, col = i // 8, i % 8
            if self.board[i] != current_player and self.board[i] != own_king:
                continue

            if self.board[i] == own_king:
                for dr, dc in [(1, -1), (1, 1), (-1, -1), (-1, 1)]:
                    step = 1
                    found_enemy = None
                    while True:
                        nr, nc = row + dr * step, col + dc * step
                        if not (0 <= nr <= 7 and 0 <= nc <= 7):
                            break
                        cell = self.board[nr * 8 + nc]
                        if found_enemy is None:
                            if cell == ' ':
                                available_moves.append((row, col, nr, nc))
                            elif cell == current_player or cell == own_king:
                                break
                            else:
                                found_enemy = (nr, nc)
                            step += 1
                        else:
                            if cell == ' ':
                                available_moves.append((row, col, nr, nc, found_enemy[0],found_enemy[1]))
                                step+=1
                            else:
                                break
            else:
                if current_player == "B":
                    if 0<=row-1<=7 and 0<=col-1<=7 and 0<=row-2<=7 and 0<=col-2<=7 and self.board[(row-1)*8+(col-1)] == opponent and self.board[(row-2)*8+(col-2)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,row-2,col-2,row-1,col-1))
                    if 0<=row-1<=7 and 0<=col-1<=7 and self.board[(row-1)*8+(col-1)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,row-1,col-1))
                    if 0<=row-1<=7 and 0<=col+1<=7 and 0<=row-2<=7 and 0<=col+2<=7 and self.board[(row-1)*8+(col+1)] == opponent and self.board[(row-2)*8+(col+2)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,row-2,col+2,row-1,col+1))
                    if 0<=row-1<=7 and 0<=col+1<=7 and self.board[(row-1)*8+(col+1)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,row-1,col+1))
                if current_player == "W":
                    if 0<=row+1<=7 and 0<=col-1<=7 and 0<=row+2<=7 and 0<=col-2<=7 and self.board[(row+1)*8+(col-1)] == opponent and self.board[(row+2)*8+(col-2)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,row+2,col-2,row+1,col-1))
                    if 0<=row+1<=7 and 0<=col-1<=7 and self.board[(row+1)*8+(col-1)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,(row+1),col-1))
                    if 0<=row+1<=7 and 0<=col+1<=7 and 0<=row+2<=7 and 0<=col+2<=7 and self.board[(row+1)*8+(col+1)] == opponent and self.board[(row+2)*8+(col+2)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,row+2,col+2,row+1,col+1))
                    if 0<=row+1<=7 and 0<=col+1<=7 and self.board[(row+1)*8+(col+1)] == ' ' and (row+col)%2==1:
                        available_moves.append((row,col,(row+1),col+1))

        return available_moves


    def make_move(self,pos):
        """
        Makes a move
        """
        if len(pos)==6:
            f_row,f_col,t_row,t_col,c_row,c_col = pos
        else:
            f_row,f_col,t_row,t_col = pos
            c_row, c_col = None, None

        from_index = f_row*8 + f_col
        to_index = t_row*8 + t_col

        was_king = self.board[from_index] in (KING_WHITE, KING_BLACK)
        captured_symbol = None
        if c_row is not None:
            captured_symbol = self.board[c_row*8 + c_col]
        self.for_king.append((was_king, captured_symbol))

        self.board[to_index] = self.board[from_index]
        self.board[from_index] = EMPTY

        if self.board[to_index] == PLAYER_WHITE and t_row == 7:
            self.board[to_index] = KING_WHITE
        elif self.board[to_index] == PLAYER_BLACK and t_row == 0:
            self.board[to_index] = KING_BLACK

        if c_row is not None:
            captured_index = c_row*8 + c_col
            self.board[captured_index] = EMPTY

        self.current_player = "B" if self.current_player == "W" else "W"
        return True

    def undo_move(self,pos):
        """ Undo the move at the given position. """
        was_king, captured_symbol = self.for_king.pop()
        if len(pos)==6:
            f_row,f_col,t_row,t_col,c_row,c_col = pos
        else:
            f_row,f_col,t_row,t_col = pos
            c_row, c_col = None, None

        from_index = f_row*8 + f_col
        to_index = t_row*8 + t_col

        moved = self.board[to_index]
        self.board[to_index] = EMPTY
        if moved in (KING_WHITE, KING_BLACK) and not was_king:
            moved = PLAYER_WHITE if moved == KING_WHITE else PLAYER_BLACK
        self.board[from_index] = moved

        if c_row is not None and captured_symbol is not None:
            c_index = c_row*8 + c_col
            self.board[c_index] = captured_symbol

        self.current_player = "B" if self.current_player == "W" else "W"
        return True

    def check_winner(self):
        """Check if the game over"""
        white_total = self.board.count(PLAYER_WHITE) + self.board.count(KING_WHITE)
        black_total = self.board.count(PLAYER_BLACK) + self.board.count(KING_BLACK)

        if white_total==0:
            return PLAYER_BLACK
        if black_total==0:
            return PLAYER_WHITE

        if not self.get_available_moves(self.current_player):
            return PLAYER_BLACK if self.current_player==PLAYER_WHITE else PLAYER_WHITE

    def is_draw(self):
        """Check if the game is draw"""
        if self.check_winner() is None and all(cell != EMPTY for cell in self.board):
            return True
        return False

    def is_game_over(self):
        """Check if the game is over"""
        return self.check_winner() is not None or self.is_draw()==True

if __name__ == "__main__":
    game = MMChecker()
    game.print_board()
