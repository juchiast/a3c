import random
import viz
from itertools import product

SPAWN_LIMIT = 70
random.seed()

class Car:
    def __init__(self, path):
        self.delay = 0
        self.path = path
        self.pos = 0
        self.time = 0

    def next(self):
        self.pos += 1
        self.time += self.delay
        self.delay = 0
        
    def wait(self):
        self.delay += 1

    def finished(self):
        return self.pos == len(self.path) - 1


class Graph:
    def read(self, file_path, display=False):
        with open(file_path, 'r') as f:
            n, m = list(map(int, f.readline().split()))
            edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
            circles = list(map(int, f.readline().split()))
            limit = int(f.readline())
        return Graph(n, edges, circles, limit, display)

    def __init__(self, n=0, edges=[], circles=[], limit=0, display=False):
        m = len(edges)
        deg = [0 for i in range(n)]
        canh = [[] for i in range(n)]

        edges_map = dict((edge, i) for i, edge in enumerate(edges))

        paths = [[[] for j in range(n)] for i in range(n)]
        for u, v in edges:
            paths[u][v].append((u, v))
            deg[v] += 1
            canh[v].append((u, v))

        for k, i, j in product(range(n), repeat=3):
            for u, v in product(paths[i][k], paths[k][j]):
                x = set(u)
                if all(y not in x for y in v[1:]):
                    paths[i][j].append(u + v[1:])

        self.n = n
        self.m = m
        self.limit = limit
        self.edges = edges
        self.circles = set(circles)
        self.paths = paths
        self.cars = []
        self.deg = deg
        self.display = display
        self.canh = canh
        self.car_count = [0 for i in range(m)]
        self.off_count = [0 for i in range(m)]
        self.edges_map = edges_map
        if display:
            self.viz = viz.Visualizer(n, edges, circles)

    def get_state(self):
        s = sum(self.car_count) + 1e-4
        off = [x for x in self.off_count]
        for i in range(len(self.car_count)):
            self.car_count[i] /= s
            off[i] = min(1, off[i] / 4)
        return self.car_count + off

    def spawn(self):
        limit = random.randint(1, SPAWN_LIMIT)
        for _ in range(limit):
            s, t = random.sample(range(self.n), 2)
            if self.paths[s][t]:
                self.cars.append(Car(random.choice(self.paths[s][t])))

    def next(self, a):
        self.spawn()

        for i in range(len(self.car_count)):
            self.car_count[i] = 0

        edges = set()
        for i, x in enumerate(a):
            if x < len(self.canh[i]):
                edges.add(self.canh[i][x])

        for i, edge in enumerate(edges):
            if edge in edges:
                self.off_count[i] = 0
            else:
                self.off_count[i] += 1
        
        moves = []
        waits = [0 for i in range(self.n)]
        count = {}
        for car in self.cars:
            a, b = car.path[car.pos-1], car.path[car.pos]
            wait = True
            if car.pos == 0 or (a, b) in edges or b in self.circles:
                s = car.path[car.pos]
                t = car.path[car.pos + 1]
                if not count.get((s, t)):
                    count[(s, t)] = 0
                if count[(s, t)] < self.limit:
                    wait = False
                    count[(s, t)] += 1
                    car.next()
                    moves.append((s, t))
                    self.car_count[self.edges_map[(s, t)]] += 1
            if wait:
                car.wait()
                waits[car.path[car.pos]] += 1
                if car.pos > 0:
                    self.car_count[self.edges_map[(a, b)]] += 1
        finished = list(filter(lambda car: car.finished(), self.cars))
        self.cars = list(filter(lambda car: not car.finished(), self.cars))
        if self.display:
            self.viz.update(edges, moves, waits)
        return (2*len(finished) - sum(waits)) / 100

if __name__ == "__main__":
    g = Graph().read('graph.txt', True)
    while True:
        g.next([0]*g.n)
