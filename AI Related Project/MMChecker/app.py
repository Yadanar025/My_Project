from flask import Flask, render_template, request, jsonify
import json
from checkers import MMChecker, PLAYER_BLACK,PLAYER_WHITE
from agent import RandomAgent, HeuristicAgent, MinimaxAgent, simulate_match

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def create_agent(agent_type, player):
    if agent_type == "random":
        return RandomAgent(player)
    elif agent_type == "heuristic":
        return HeuristicAgent(player)
    else:
        return MinimaxAgent(player)

#send the data to the server and get the responses
@app.route("/play", methods=["POST"])
def play():
    data = request.get_json()
    board = data.get("board")
    current_player = data.get("current_player")
    game = MMChecker(board=board, current_player=current_player)
    game.make_move(data.get("move"))
    over=game.is_game_over()
    over=game.is_game_over()
    move_scores = None
    nodes_explored = None
    if not over:
        agent_type = data.get("agent_type")
        if agent_type:
            agent_player = PLAYER_BLACK if current_player == PLAYER_WHITE else PLAYER_WHITE
            agent = create_agent(agent_type, agent_player)
            agent_move = agent.get_move(game)
            game.make_move(agent_move)
            move_scores = getattr(agent, 'last_move_scores', None)
            nodes_explored = getattr(agent, 'nodes_explored', None)
    over=game.is_game_over()
    return jsonify({"board": game.board,
                "current_player": game.current_player,
                "winner": game.check_winner(),
                "available_moves": game.get_available_moves(game.current_player),
                "game_over": over,
                "move_scores": move_scores,
                "nodes_explored": nodes_explored})


@app.route("/legal_moves", methods=["POST"])
def legal_moves():
    data = request.get_json()
    board = data.get("board")
    current_player = data.get("current_player")
    game = MMChecker(board=board, current_player=current_player)
    return jsonify({"available_moves": game.get_available_moves(current_player)})

@app.route("/ai_move", methods=["POST"])
def ai_move():
    data = request.get_json()
    board = data.get("board")
    current_player = data.get("current_player")
    game = MMChecker(board=board, current_player=current_player)
    over=game.is_game_over()
    if not over:
        agent_x_type = data.get("agent_x_type")
        agent_o_type = data.get("agent_o_type")
        agent_type = agent_x_type if current_player == PLAYER_WHITE else agent_o_type
        agent = create_agent(agent_type, current_player)
        agent_move = agent.get_move(game)
        game.make_move(agent_move)
        over=game.is_game_over()
    return jsonify({"board": game.board, 
                    "current_player": game.current_player,
                    "winner": game.check_winner(), 
                    "available_moves": game.get_available_moves(game.current_player),
                    "game_over": over})
@app.route("/benchmark", methods=["POST"])
def benchmark():
    data = request.get_json()
    agent1_type = data.get("agent1_type")
    agent2_type = data.get("agent2_type")
    num_games = data.get("num_games", 50)
    agent1 = create_agent(agent1_type, PLAYER_WHITE)
    agent2 = create_agent(agent2_type, PLAYER_BLACK)
    results = simulate_match(agent1, agent2, num_games)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)