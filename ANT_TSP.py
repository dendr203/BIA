import numpy as np
import plotly.graph_objects as go


class TSP:
    def __init__(self, cities, n_ants=50, n_iterations=200, alpha=1, beta=5, rho=0.5, visualize_afterN = 5):
        self.cities = cities
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.visualize_afterN = visualize_afterN
        
        self.n = len(cities)
        self.dist_matrix = self.compute_distance_matrix()
        self.pheromone = np.ones((self.n, self.n)) #počáteční feromon
        self.best_route = None
        self.best_distance = float("inf")

    # Vytvoří 2D pole vzdáleností mezi městy pomocí Eukleidovské normy
    def compute_distance_matrix(self):
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dist_matrix[i, j] = np.linalg.norm(np.array(self.cities[i]) - np.array(self.cities[j]))
        return dist_matrix

    
    def construct_solution(self):
        solutions = []
        
        for _ in range(self.n_ants):
            # pro každého mravence si vyberu random počáteční město
            visited = [np.random.randint(self.n)]
            
            while len(visited) < self.n:
                current = visited[-1]
                probabilities = []
                
                # výpočet pravděpodobnosti přechodu
                for j in range(self.n):
                    if j not in visited:
                        # pro nenavštívené město j se spočítá pravděpodobnost
                        tau = self.pheromone[current][j] ** self.alpha              #síla feromonu mezi městy (current -> j)
                        eta = (1 / self.dist_matrix[current][j]) ** self.beta       #viditelnost - inverzní vzdálenost neboli heuristika (1/vzdálenost)
                        probabilities.append((j, tau * eta))
                
                # součet všech atraktivit
                total = sum(p for _, p in probabilities)
                # normalizace - dostaneme pravděpodobnostní rozdělení = 1
                probabilities = [(city, p / total) for city, p in probabilities]
                # výběr dalšího města náhodně z vypočtených pravděpodobností (více feromonu/kratší vzdálenost => větší pravděpodobnost => větší šance na vybrání)
                next_city = np.random.choice([city for city, _ in probabilities],
                                            p=[p for _, p in probabilities])
                visited.append(next_city)
            solutions.append(visited)
        return solutions

    
    def update_feromones(self, solutions):
        self.pheromone *= (1 - self.rho)  # evaporace

        
        for route in solutions:
            # délka cesty mravence
            distance = sum(self.dist_matrix[route[i]][route[(i + 1) % self.n]] for i in range(self.n))
            # průchod městy a ohodnocení hran novými feromony
            for i in range(self.n):
                a, b = route[i], route[(i + 1) % self.n]
                self.pheromone[a][b] += 1 / (distance + 1e-6)
                self.pheromone[b][a] += 1 / (distance + 1e-6)  # symetrie
                # lepší kratší cesty přidají více feromonů

    def run(self):
        best_distances = []
        # vytvoření počátečních cest pro první vizualizaci
        initial_route = list(np.random.permutation(self.n))
        initial_distance = sum(self.dist_matrix[initial_route[i]][initial_route[(i + 1) % self.n]] for i in range(self.n))
        best_generation = 0
        
        self.visualize(initial_route, initial_distance, best_generation)
        
        for iteration in range(1, self.n_iterations + 1):
            solutions = self.construct_solution()
            self.update_feromones(solutions)

            # prozkoumáme nový solution jestli tam daný mravenec nepřidal lepší který bude globální best
            for route in solutions:
                distance = sum(self.dist_matrix[route[i]][route[(i + 1) % self.n]] for i in range(self.n))
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route.copy()
                    best_generation = iteration

            best_distances.append(self.best_distance)
            
            if iteration != 0 and iteration % self.visualize_afterN == 0:
                self.visualize(self.best_route, self.best_distance, generation=iteration)

        return self.best_route, self.best_distance, best_generation, best_distances


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


def ANT_for_TSP():
    print("Start ANT for TSP")
    n_cities = 50
    cities = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n_cities)]
    #alpha - váha feromonu pří výberu cesty (větší = silnější vliv kolektivní paměti)
    #beta - váha heuristiky (větší = silnější vliv lokální informace)
    #rho - míra evaporace feromonu (větší = rychlejší zapomínání, tedy menší riziko uváznutí v lokálním minimu)
    ant = TSP(cities, n_ants=10, n_iterations=30, alpha=1, beta=5, rho=0.5, visualize_afterN=5)

    best_route, best_distance, best_generation, best_distances = ant.run()
    print(f"Best distance found: {best_distance}")

    # Final visualization of the best route
    ant.visualize(best_route, best_distance, best_generation, best_distances)
    return


if __name__ == "__main__":
    ANT_for_TSP()
