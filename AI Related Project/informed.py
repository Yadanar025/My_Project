

def greedy_best_first_search(data, start, goal, direction):
    visited = set()
    queue = [(start, [start])]
    while queue:
        queue.sort(key=lambda x: direction(x[0], goal))
        current_node, path = queue.pop(0)
        if current_node == goal:
            return path
        visited.add(current_node)
        for note2, edge_cost in data[current_node]:
            if note2 not in visited:
                new_path = path + [note2]
                queue.append((note2, new_path))
    return None

def A_star_search(data, start, goal, direction):
    visited = set()
    queue = [(start, [start], 0)]
    while queue:
        queue.sort(key=lambda x: x[2] + direction(x[0], goal))
        current_node, path, cost = queue.pop(0)
        if current_node == goal:
            return path
        visited.add(current_node)
        for note2, edge_cost in data[current_node]:
            if note2 not in visited:
                new_path = path + [note2]
                new_cost = cost + edge_cost
                queue.append((note2, new_path, new_cost))
    return None

if __name__ == "__main__":
        
    data1 = {
    "S": [("A", 1), ("B", 3),("D",5)],
    "A": [("C", 4)],
    "B": [("D", 4)],
    "C": [("D", 4),("G", 2)],
    "D": [("G", 5)],
    "G": []
    }

    print(greedy_best_first_search(data1, 'S', 'G', lambda x, y: 0))
    print(A_star_search(data1, 'S', 'G', lambda x, y: 0))