import numpy as np
import plotly.graph_objects as go
import time

class TSP:
    def __init__(self, cities, population_size=100, generations=200, mutation_rate=0.02):
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.n = len(cities)

        self.dist_matrix = self.compute_distance_matrix()

    def compute_distance_matrix(self):
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dist_matrix[i, j] = np.linalg.norm(np.array(self.cities[i]) - np.array(self.cities[j]))
        return dist_matrix

    def route_distance(self, route):
        return sum(
            self.dist_matrix[route[i]][route[(i + 1) % self.n]] for i in range(self.n)
        )

    def fitness(self, population):
        return [1 / (self.route_distance(route) + 1e-6) for route in population]

    def initial_population(self):
        return [np.random.permutation(self.n).tolist() for _ in range(self.population_size)]

    def select_parents(self, population, fitness_scores):
        parents = []
        for _ in range(self.population_size):
            candidates = np.random.choice(self.population_size, 3)
            best = max(candidates, key=lambda x: fitness_scores[x])
            parents.append(population[best])
        return parents

    def crossover(self, parent1, parent2):
        n = self.n
        start, end = sorted(np.random.choice(n, 2))
        child = [-1] * n
        child[start:end] = parent1[start:end]

        p2_elements = [item for item in parent2 if item not in child[start:end]]

        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = p2_elements[idx]
                idx += 1

        return child

    def mutate(self, route):
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(self.n, 2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route

    def run(self, visualize_process=False):
        population = self.initial_population()
        best_route = None
        best_distance = float('inf')

        for gen in range(self.generations):
            fitness_scores = self.fitness(population)
            parents = self.select_parents(population, fitness_scores)
            next_population = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % self.population_size]
                child1 = self.crossover(p1, p2)
                child2 = self.crossover(p2, p1)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            population = next_population[: self.population_size]

            distances = [self.route_distance(route) for route in population]
            min_idx = np.argmin(distances)
            if distances[min_idx] < best_distance:
                best_distance = distances[min_idx]
                best_route = population[min_idx].copy()

            if visualize_process:
                self.visualize(population[min_idx], best_distance, generation=gen)
                time.sleep(0.05)  # Pause for visualization

        return best_route, best_distance

    def visualize(self, route, distance, generation=None):
        x = [self.cities[i][0] for i in route] + [self.cities[route[0]][0]]
        y = [self.cities[i][1] for i in route] + [self.cities[route[0]][1]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                marker=dict(size=8, color='blue'),
                line=dict(color='red', width=2),
                name=f'Route, distance={distance:.2f}'
            )
        )

        title = 'GA for Travelling Salesman Problem'
        if generation is not None:
            title += f' - Generation {generation}'

        fig.update_layout(
            title=title,
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            height=600,
            width=800
        )
        fig.show(renderer="browser")


if __name__ == '__main__':
    print('Start GA for TSP')
    n_cities = 30
    cities = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n_cities)]

    tsp = TSP(cities, population_size=100, generations=200, mutation_rate=0.02)

    # Run GA with visualization of the search process
    best_route, best_distance = tsp.run(visualize_process=False)
    print(f'Best distance found: {best_distance}')

    # Final visualization of the best route
    tsp.visualize(best_route, best_distance)
