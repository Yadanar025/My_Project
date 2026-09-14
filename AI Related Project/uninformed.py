import queue


def bfs(data, start, goal,visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    queue = [start]
    parent = {}


    while queue:
        current_node = queue.pop(0)
        for i,edge_cost in data[current_node]:
            if i not in visited:
                visited.add(i)
                parent[i]= current_node
                queue.append(i)
                if i == goal:
                    return reconstruct_path(parent, start, goal)
        


def dfs(data, start, goal, visited=None, parent=None):
    if parent is None:
        parent = {}
    if visited is None:
        visited = set()

    visited.add(start)
    if start != goal:
        
            for i,edge_cost in data[start]:
                if i not in visited:
                    visited.add(i)
                    parent[i]= start
                    if i==goal:
                        return reconstruct_path(parent, start, goal)
                    result = dfs(data, i, goal, visited, parent)
                    if result:
                        return result

def reconstruct_path(parent, start, goal):
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def ucs(data, start, goal):
    visited = []
    queue = [[(start,0)]]
    while queue:
        queue.sort(key=lambda x: sum(c for _, c in x))
        direction = queue.pop(0)
        current_node = direction[-1][0]
        if current_node in visited:
            continue
        else:
            nodes = data.get(current_node, [])
            visited.append(current_node)
            if current_node == goal:
                total_cost = sum(c for _, c in direction)
                return total_cost, [node for node, _ in direction]
            for (node2, edge_cost) in nodes:
                new_path = direction.copy()
                new_path.append((node2, edge_cost))
                queue.append(new_path) 

def ids(data,start,goal,depth):
    
    for i in range(depth):
        if depth <=0:
            return None
        result =limited_dfs(data,start,goal,i)
        if result is not None:
            return result
    return None

def limited_dfs(data, start, goal, depth,visited=None, parent=None):
    if visited is None:
        visited = set()
    if parent is None:
        parent = {}
    visited.add(start)
    for i,edge_cost in data[start]:
        if depth <=0:
            return None
        if i not in visited:
            visited.add(i)
            parent[i]= start
            if i==goal:
                return reconstruct_path(parent, start, goal)
            result = limited_dfs(data, i, goal, depth - 1, visited, parent)
            if result:
                return result
            else:
                visited.remove(i)
                del parent[i]
    return None


if __name__ == "__main__":
    data = {
        '0': {'1', '2', '3'},
        '1': {'0', '2'},
        '2': {'0', '1','4'},
        '3': {'0'},
        '4': {'2'},
    }
    #data = {'S':[(1,'A'),(12,'G')]}
    data1 = {
        "S": [("A", 1), ("B", 3),("D",5)],
        "A": [("C", 4)],
        "B": [("D", 4)],
        "C": [("D", 4),("G", 2)],
        "D": [("G", 5)],
        "G": []
    }

    print(ids(data,'0','4',5))
    #print(ucs(data1, 'S', 'G'))
