import numpy as np
import plotly.graph_objects as go


class TSP:
    def __init__(self, cities, population_size=100, generations=200, mutation_rate=0.02):
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.n = len(cities)

        self.dist_matrix = self.compute_distance_matrix()

    # Vytvoří 2D pole vzdáleností mezi městy pomocí Eukleidovské normy
    def compute_distance_matrix(self):
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dist_matrix[i, j] = np.linalg.norm(np.array(self.cities[i]) - np.array(self.cities[j]))
        return dist_matrix

    # Spočítá celkovou délku dané cesty (včetně návratu do výchozího města)
    def route_distance(self, route):
        return sum(self.dist_matrix[route[i]][route[(i + 1) % self.n]] for i in range(self.n))

    # Převrácená hodnota vzdálenosti - čím kratší cesta, tím vyšší fittnes
    def fitness(self, population):
        return [1 / (self.route_distance(route) + 1e-6) for route in population]

    # Generace náhodné permutace měst jako počáteční stav
    def initial_population(self):
        return [np.random.permutation(self.n).tolist() for _ in range(self.population_size)]

    # Ze 3 náhodných jedinců vybere toho s nejlepším fitness
    def select_parents(self, population, fitness_scores):
        parents = []
        for _ in range(self.population_size):
            candidates = np.random.choice(self.population_size, 3)
            best = max(candidates, key=lambda x: fitness_scores[x])
            parents.append(population[best])
        return parents

    # Částečně zachová segment z jednoho rodiče a doplní z druhého bez duplicit
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

    # S pravděpodobností mutation_rate prohodí dvě města v cestě
    def mutate(self, route):
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(self.n, 2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route

    def run(self):
        population = self.initial_population()
        best_route = population[0]
        best_distance = self.route_distance(best_route)
        best_generation = 0
        best_distances = []

        self.visualize(best_route, best_distance, best_generation)

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
                best_generation = gen

            best_distances.append(best_distance)

        return best_route, best_distance, best_generation, best_distances

    def visualize(self, route, distance, generation=None, best_distances=None):
        x = [self.cities[i][0] for i in route] + [self.cities[route[0]][0]]
        y = [self.cities[i][1] for i in route] + [self.cities[route[0]][1]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers+text",
                marker=dict(size=8, color="blue"),
                line=dict(color="red", width=2),
                text=[str(i) for i in route] + [str(route[0])],
                textposition="top center",
                name=f"Route, distance={distance:.2f}",
            )
        )

        title = "GA for Travelling Salesman Problem"
        if generation is not None:
            title += f" - Generation {generation}"

        fig.update_layout(
            title=title,
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            height=600,
            width=800,
        )
        fig.show(renderer="browser")

        if best_distances is not None:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(y=best_distances, mode="lines", line=dict(color="green"))
            )
            fig.update_layout(
                title="Vývoj nejlepší vzdálenosti",
                xaxis_title="Generace",
                yaxis_title="Vzdálenost",
            )
            fig.show(renderer="browser")


def GA_for_TSP():
    print("Start GA for TSP")
    n_cities = 20

    cities = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n_cities)]

    tsp = TSP(cities, population_size=100, generations=200, mutation_rate=0.02)

    # Run GA with visualization of the search process
    best_route, best_distance, best_generation, best_distances = tsp.run()
    print(f"Best distance found: {best_distance}")

    # Final visualization of the best route
    tsp.visualize(best_route, best_distance, best_generation, best_distances)
    return


if __name__ == "__main__":
    GA_for_TSP()
