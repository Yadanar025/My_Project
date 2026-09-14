from flask import Flask, render_template, request, jsonify
import json
from uninformed import bfs,dfs,ucs,ids
from informed import greedy_best_first_search, A_star_search
app = Flask(__name__)
with open("map_data.json", "r") as f:
    map_data=json.load(f)

graph = map_data["graph"]
coordinates = map_data["coordinates"]

import math
def heuristic(start,goal):
    latstart,lonstart = coordinates[start]
    latgoal,longoal = coordinates[goal]
    return math.dist((latstart,lonstart),(latgoal,longoal))

def path_cost(path):
    total = 0
    for i in range(len(path)-1):
        for neighbor, count in graph[path[i]]:
            if neighbor == path[i+1]:
                total += count
                break
    return total


@app.route("/")
def index():
    return render_template("index.html",cities=list(graph.keys()),coordinates=coordinates)

@app.route("/search", methods=["POST"])
def click():
    requested_algorithm = request.form.get("algorithm")
    start_city = request.form.get("start_city")
    goal_city = request.form.get("distination_city")
    cost,path =search_algorithm(requested_algorithm, start_city, goal_city)
    result = {
        "algorithm": requested_algorithm,
        "start_city": start_city,
        "goal_city": goal_city,
        "cost": cost,
        "path": path}
    return jsonify(result)

def search_algorithm(algorithm,start,goal):
    if algorithm=="BFS":
        path = bfs(graph,start,goal)
        return path_cost(path), path
    elif algorithm=="DFS":
        path = dfs(graph,start,goal)
        return path_cost(path), path
    elif algorithm == "UCS":
        return ucs(graph,start,goal)
    elif algorithm == "IDS":
        path = ids(graph,start,goal,depth=len(graph))
        return path_cost(path), path
    elif algorithm == "Greedy Best First Search":
        path = greedy_best_first_search(graph,start,goal,heuristic)
        return path_cost(path), path
    elif algorithm == "A* Search":
        path = A_star_search(graph,start,goal,heuristic)
        return path_cost(path), path


if __name__ == "__main__":
    app.run(debug=True)
 
