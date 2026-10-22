from checkers import MMChecker, PLAYER_WHITE, PLAYER_BLACK, KING_WHITE, KING_BLACK

class RandomAgent:
    def __init__(self, player=None):
        self.player = player
    
    def get_move(self,checkers):
        import random
        return random.choice(checkers.get_available_moves(checkers.current_player))
    
class HeuristicAgent:
    def __init__(self, player=None):
        self.player = player

    def get_move(self, checkers):
        best_score = float('-inf')
        best_move = None
        for move in checkers.get_available_moves(checkers.current_player):
            checkers.make_move(move)
            score = self.evaluate_board(checkers)
            checkers.undo_move(move)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def evaluate_board(self, checkers):
        opponent = PLAYER_BLACK if self.player == PLAYER_WHITE else PLAYER_WHITE
        my_king = KING_WHITE if self.player == PLAYER_WHITE else KING_BLACK
        opp_king = KING_BLACK if self.player == PLAYER_WHITE else KING_WHITE

        my_pieces = checkers.board.count(self.player)
        opp_pieces = checkers.board.count(opponent)
        my_kings = checkers.board.count(my_king)
        opp_kings = checkers.board.count(opp_king)

        my_chance = 0
        opp_chance = 0
        for i in range(len(checkers.board)):
            row = i // 8
            if checkers.board[i] == self.player:
                my_chance += row if self.player == PLAYER_WHITE else (7 - row)
            elif checkers.board[i] == opponent:
                opp_chance += row if opponent == PLAYER_WHITE else (7 - row)

        KING_WEIGHT = 3
        ADVANCEMENT_WEIGHT = 0.1

        score = (my_pieces - opp_pieces) \
            + KING_WEIGHT * (my_kings - opp_kings) \
            + ADVANCEMENT_WEIGHT * (my_chance - opp_chance)
        return score



class MinimaxAgent:
    def __init__(self, player=PLAYER_WHITE,use_alpha_beta=True,max_depth=4):
        self.max_depth = max_depth
        self.player = player
        self.opponent_mark = PLAYER_WHITE if self.player == PLAYER_BLACK else PLAYER_BLACK
        self.use_alpha_beta = use_alpha_beta
        self.nodes_explored = 0
        self.last_move_scores = {}      


    def get_move(self, checkers):
        self.nodes_explored = 0
        self.last_move_scores = {}
        best_score = float('-inf')
        best_move = None
        for move in checkers.get_available_moves(checkers.current_player):
            checkers.make_move(move)
            score = self.minimax(checkers, 0, False, float('-inf'), float('inf'))
            checkers.undo_move(move)
            self.last_move_scores[move] = score
            if score > best_score:
                best_score = score
                best_move = move
        return best_move


    def evaluate_board(self, checkers):
            opponent = PLAYER_BLACK if self.player == PLAYER_WHITE else PLAYER_WHITE
            my_king = KING_WHITE if self.player == PLAYER_WHITE else KING_BLACK
            opp_king = KING_BLACK if self.player == PLAYER_WHITE else KING_WHITE
    
            my_pieces = checkers.board.count(self.player)
            opp_pieces = checkers.board.count(opponent)
            my_kings = checkers.board.count(my_king)
            opp_kings = checkers.board.count(opp_king)
    
            my_chance = 0
            opp_chance = 0
            for i in range(len(checkers.board)):
                row = i // 8
                if checkers.board[i] == self.player:
                    my_chance += row if self.player == PLAYER_WHITE else (7 - row)
                elif checkers.board[i] == opponent:
                    opp_chance += row if opponent == PLAYER_WHITE else (7 - row)
    
            KING_WEIGHT = 3
            ADVANCEMENT_WEIGHT = 0.1
    
            score = (my_pieces - opp_pieces) \
                + KING_WEIGHT * (my_kings - opp_kings) \
                + ADVANCEMENT_WEIGHT * (my_chance - opp_chance)
            return score
    
    def minimax(self, checkers, depth, maximizing, alpha=float('-inf'), beta=float('inf')):
        self.nodes_explored += 1
        winner = checkers.check_winner()
        if winner == self.player:
            return 1000 - depth
        elif winner == self.opponent_mark:
            return depth - 1000
        elif not checkers.get_available_moves(checkers.current_player):
            return 0
        elif depth >= self.max_depth:
            return self.evaluate_board(checkers)

        if maximizing:
            best_score = float('-inf')
            for move in checkers.get_available_moves(checkers.current_player):
                checkers.make_move(move)
                score = self.minimax(checkers, depth + 1, False, alpha, beta)
                checkers.undo_move(move)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if self.use_alpha_beta and alpha >= beta:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in checkers.get_available_moves(checkers.current_player):
                checkers.make_move(move)
                score = self.minimax(checkers, depth + 1, True, alpha, beta)
                checkers.undo_move(move)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if self.use_alpha_beta and alpha >= beta:
                    break
            return best_score


def simulate_match(agent1,agent2, num_games=50):
    results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
    for _ in range(num_games):
        checkers = MMChecker()
        current_agent = agent1 if checkers.current_player == agent1.player else agent2
        while True:
            move = current_agent.get_move(checkers)
            checkers.make_move(move)
            winner = checkers.check_winner()
            if winner==agent1.player:
                results['agent1_wins'] += 1
                break
            elif winner==agent2.player:
                results['agent2_wins'] += 1
                break    
            elif not checkers.get_available_moves(checkers.current_player):
                results['draws'] += 1
                break

            current_agent = agent1 if current_agent == agent2 else agent2
    return results

