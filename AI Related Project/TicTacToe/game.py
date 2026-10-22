PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

class TicTacToe:

    def __init__(self,board=None,current_player=None):
        if board is None:
            self.board = [' ' for _ in range(9)]
        else:
            self.board = board
        if current_player is None:
            self.current_player = PLAYER_X
        else:
            self.current_player = current_player


    def get_available_moves(self):
        """
        Returns a list of available moves on the board.
        """
        available_moves = []
        for i in range(len(self.board)):
            if self.board[i] == ' ':
                available_moves.append(i)
        return available_moves

    def is_valid_move(self,pos):
        """
        Checks validity of a move at the given position.
        """
        if self.board[pos] == ' ':
            return True
        return False

    def make_move(self,pos):
        """
        Makes a move
        """
        if not self.is_valid_move(pos):
            print("Invalid move") 
            return False
        else:
            self.board[pos] = self.current_player
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
            return True        

    
    def undo_move(self,pos):
        """ Undo the move at the given position. """
        if self.board[pos] != EMPTY:
            self.current_player = self.board[pos]
            self.board[pos] = EMPTY

    def check_winner(self):
        """Check if there's a winner on the board."""
        winning_line = self.get_winning_line()
        if not winning_line:
            return None
        return self.board[winning_line[0]]
 
    def get_winning_line(self):
        winning_possibility = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for pos in winning_possibility:
            if self.board[pos[0]]==self.board[pos[1]]==self.board[pos[2]] and self.board[pos[0]] != EMPTY:
                return pos
        return None

    def is_draw(self):
        """Check if the game is draw"""
        if self.check_winner() is None and all(cell != EMPTY for cell in self.board):
            return True
        return False

    def is_game_over(self):
        """Check if the game is over"""
        return self.check_winner() is not None or self.is_draw()==True