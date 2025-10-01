import os
import numpy as np

import plotly.graph_objects as go

resolution = 500
graph_folder_name = "graphs_files"


class FunctionVisualizer:
    def __init__(
        self, func_name, dimensions, lower_bound, upper_bound, make_new_file=False
    ):
        self.name = func_name
        self.d = dimensions
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound

        self.mk = make_new_file

    # FUNCTIONS
    # každý bod má již počet dimenzí, takže můžu použít jen sum a nemusím specifikovat dimenze
    def sphere(self, x):
        return np.sum(x**2)

    def ackley(self, x, a=20, b=0.2, c=np.pi * 2):
        sum_sqrt = np.sum(np.square(x))
        sum_cos = np.sum(np.cos(c * x))

        first_term = -a * np.exp(-b * (np.sqrt(sum_sqrt / self.d)))
        second_term = -np.exp(sum_cos / self.d)

        return first_term + second_term + a + np.exp(1)

    def rastrigin(self, x):
        sum_bracket = np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))

        return 10 * self.d + sum_bracket

    def rosenbrock(self, x):
        return np.sum(
            100 * np.square(x[1:] - np.square(x[:-1])) + (np.square(x[:-1] - 1))
        )

    def griewank(self, x):
        sum = np.sum(np.square(x) / 4000)

        prod = 1
        for i in range(self.d):
            prod *= np.cos(x[i] / (np.sqrt(i + 1)))

        return sum - prod + 1

    def schwefel(self, x):
        return 418.9829 * self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def levy(self, x):
        w = 1 + ((x - 1) / 4.0)

        first_term = np.square(np.sin(np.pi * w[0]))
        if self.d > 1:
            sum_term = np.sum(
                np.square(w[:-1] - 1) * (1 + 10 * np.square(np.sin(np.pi * w[:-1] + 1)))
            )
        else:
            sum_term = 0.0

        second_term = np.square(w[-1] - 1) * (1 + np.square(np.sin(2 * np.pi * w[-1])))

        return first_term + sum_term + second_term

    def michalewicz(self, x, m=10):
        i = np.arange(1, self.d + 1)

        return -np.sum(np.sin(x) * np.sin((i * np.square(x)) / (np.pi)) ** (2 * m))

    def zakharov(self, x):
        i = np.arange(1, self.d + 1)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * i * x)
        return sum1 + sum2**2 + sum2**4

    # FUNCTIONS END

    # eval what functions we want to use in visualizing
    def function_type(self, x):
        if self.name == "sphere":
            return self.sphere(x)
        elif self.name == "ackley":
            return self.ackley(x)
        elif self.name == "rastrigin":
            return self.rastrigin(x)
        elif self.name == "rosenbrock":
            return self.rosenbrock(x)
        elif self.name == "griewank":
            return self.griewank(x)
        elif self.name == "schwefel":
            return self.schwefel(x)
        elif self.name == "levy":
            return self.levy(x)
        elif self.name == "michalewicz":
            return self.michalewicz(x)
        elif self.name == "zakharov":
            return self.zakharov(x)

    def create_graph_file(self):
        # vytvoření navzájem si stejně vzdálených bodů na ose x a y s daným rozlišením (teda kolik bodů chci mít na jedné ose)
        # np.linspace využito protože vždy bude obsahovat koncový bod (vždy se dopočítávají ostatní body aby se vešel)
        x = np.linspace(self.lB, self.uB, resolution)
        y = np.linspace(self.lB, self.uB, resolution)
        # vytvoření mřížky všech kombinací x_i y_j z těchto hodnot
        X, Y = np.meshgrid(x, y)

        # vytvoření pole pro Z souřadnice stejného tvaru jako X[d]
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.zeros(self.d)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                Z[i, j] = self.function_type(point)
        return X, Y, Z

    def visualise(self, X, Y, Z, label, highlight_points=None, history_temp=None, best_point=None):
        fig = go.Figure()

        # Přidání povrchu (surface)
        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, opacity=1, colorscale="jet", showscale=False)
        )

        # Přidání scatter bodů, pokud je předám
        if highlight_points is not None:
            x_vals = highlight_points[0]
            y_vals = highlight_points[1]
            z_vals = highlight_points[2]
            
            if history_temp is not None:
                fig.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers',
                    marker=dict(size=5, color=history_temp, colorscale='Hot', colorbar=dict(title='Teplota')),
                    name=label
                ))
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode="lines+markers",
                        marker=dict(size=5, color="red"),
                        line=dict(color="red", width=4),
                        name=label
                    )
                )
        
        if best_point is not None:
            bx, by, bf = best_point
            fig.add_trace(go.Scatter3d(
                x=[bx], y=[by], z=[bf],
                mode="markers",
                marker=dict(size=8, color="lime", symbol="diamond"),
                name="Best point"
        ))

        # Nastavení layoutu (titulek, osy, pohled)
        fig.update_layout(
            title=f"{self.name.capitalize()} function",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="f(x)",
                xaxis=dict(range=[self.lB, self.uB]),
                yaxis=dict(range=[self.lB, self.uB]),
                zaxis=dict(range=[Z.min(), Z.max()]),
                aspectmode="cube",  # Zachová proporce
                camera=dict(eye=dict(x=-1.2, y=-1.2, z=0.8)),
            ),
            margin=dict(l=80, r=80, t=100, b=80),
        )
        fig.show(renderer="browser")
        return

    # compute graphs beforehand or get them from file
    def compute(self):
        os.makedirs(graph_folder_name, exist_ok=True)
        filepath = os.path.join(graph_folder_name, f"{self.name}_data.npz")

        if os.path.exists(filepath) and not self.mk:
            print(f"Loading cached data from existing file {self.name}_data.npz")
            data = np.load(filepath)
            X = data["X"]
            Y = data["Y"]
            Z = data["Z"]
        else:
            print(f"Data not found, computing them and saving to {self.name}_data.npz")
            X, Y, Z = self.create_graph_file()
            np.savez(filepath, X=X, Y=Y, Z=Z)
            print("Data saved!!!")

        return X, Y, Z

    # ALGORITHMS
    def blind_search(self, iterations=50):
        X, Y, Z = self.compute()

        best_x = None
        best_f = np.inf
        for _ in range(iterations):
            candidate = np.random.uniform(self.lB, self.uB, self.d)
            f_val = self.function_type(candidate)
            if f_val < best_f:
                best_f = f_val
                best_x = candidate

        coordinates = np.array([[best_x[0]], [best_x[1]], [best_f]])

        self.visualise(X, Y, Z, "Blind search algorithm", coordinates)

        return

    def hill_climb(self, neighbours=10, steps=100, step_size=0.1):
        X, Y, Z = self.compute()

        # Inicializace náhodného startu (podle dimenze udělá current a current_value z current)
        current = np.random.uniform(self.lB, self.uB, self.d)
        current_value = self.function_type(current)

        # Seznamy pro ukládání historie
        history_x = [current[0]]
        history_y = [current[1]]
        history_f = [current_value]

        for _ in range(steps):
            best_candidate = current
            best_value = current_value

            for _ in range(neighbours):
                # Vytvoření souseda s náhodnou odchylkou
                candidate = current + np.random.uniform(-step_size, step_size, self.d)
                candidate = np.clip(candidate, self.lB, self.uB)  # udržení v mezích
                value = self.function_type(candidate)

                if value < best_value:
                    best_candidate = candidate
                    best_value = value

            # Pokud se zlepšilo, posuň se
            if best_value < current_value:
                current = best_candidate
                current_value = best_value
                history_x.append(current[0])
                history_y.append(current[1])
                history_f.append(current_value)
            else:
                break  # žádné zlepšení → konec

        print(f"Hill climb result: {current}, f_value: {current_value}")
        history_coord = np.array([history_x, history_y, history_f])
        self.visualise(X, Y, Z, "Hill climb algorithm", history_coord)
        return

    def simulated_annealing(self, initial_temp=100, final_temp=1, alpha=0.95, steps_per_temp=10, sigma=1.0):
        X, Y, Z = self.compute()

        # vytvoření náhodného startovacího bodu v rozmezí funkce
        current = np.random.uniform(self.lB, self.uB, self.d)
        current_value = self.function_type(current)

        # tento start je zároveň nejlepší při začátku
        best = current.copy()
        best_value = current_value

        # inicializace teploty a historie
        temp = initial_temp
        history_temp = [initial_temp]

        history_x = [current[0]]
        history_y = [current[1]]
        history_f = [current_value]

        # hlavní smyčka ochlazování
        while temp > final_temp:
            # pro každý krok teploty proveď několik iterací
            for _ in range(steps_per_temp):
                # vytvoření kandidáta od current s náhodnou odchylkou závislou na sigmě
                candidate = current + np.random.normal(0, sigma, self.d)
                candidate = np.clip(candidate, self.lB, self.uB)
                value = self.function_type(candidate)

                delta = value - current_value

                # Přijmi lepší (funkční hodnotu kandidáta) nebo horší s pravděpodobností podle Boltzmannovy distribuce
                if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                    current = candidate
                    current_value = value

                    history_x.append(current[0])
                    history_y.append(current[1])
                    history_f.append(current_value)
                    history_temp.append(temp)

                    # aktualizace nejlepšího řešení, pokud je lepší jak dosavadní nejlepší
                    if current_value < best_value:
                        best = current.copy()
                        best_value = current_value
            temp *= alpha  # ochlazení
            
        print(f"Simulated annealing result: {best}, f_value: {best_value}")
        history_coord = np.array([history_x, history_y, history_f])
        self.visualise(X, Y, Z, "Simulated annealing", history_coord, history_temp, best_point=(best[0], best[1], best_value))
        return

    # ALGORITHMS END


# ALGORITHMS MASS CALL
def blind_search_all():
    sphere = FunctionVisualizer("sphere", 2, -5.12, 5.12)
    sphere.blind_search()
    ackley = FunctionVisualizer("ackley", 2, -32.768, 32.768)
    ackley.blind_search()
    rastrigin = FunctionVisualizer("rastrigin", 2, -5.12, 5.12)
    rastrigin.blind_search()
    rosenbrock = FunctionVisualizer("rosenbrock", 2, -2.048, 2.048)
    rosenbrock.blind_search()
    griewank = FunctionVisualizer("griewank", 2, -10, 10)
    griewank.blind_search()
    schwefel = FunctionVisualizer("schwefel", 2, -500, 500)
    schwefel.blind_search()
    levy = FunctionVisualizer("levy", 2, -10, 10)
    levy.blind_search()
    michalewicz = FunctionVisualizer("michalewicz", 2, 0, np.pi)
    michalewicz.blind_search()
    zakharov = FunctionVisualizer("zakharov", 2, -10, 10)
    zakharov.blind_search()


def hill_climb_all():
    sphere = FunctionVisualizer("sphere", 2, -5.12, 5.12)
    sphere.hill_climb()
    ackley = FunctionVisualizer("ackley", 2, -32.768, 32.768)
    ackley.hill_climb()
    rastrigin = FunctionVisualizer("rastrigin", 2, -5.12, 5.12)
    rastrigin.hill_climb()
    rosenbrock = FunctionVisualizer("rosenbrock", 2, -2.048, 2.048)
    rosenbrock.hill_climb()
    # zoom, normal -600, 600
    griewank = FunctionVisualizer("griewank", 2, -10, 10)
    griewank.hill_climb()
    schwefel = FunctionVisualizer("schwefel", 2, -500, 500)
    schwefel.hill_climb()
    levy = FunctionVisualizer("levy", 2, -10, 10)
    levy.hill_climb()
    michalewicz = FunctionVisualizer("michalewicz", 2, 0, np.pi)
    michalewicz.hill_climb()
    zakharov = FunctionVisualizer("zakharov", 2, -10, 10)
    zakharov.hill_climb()


def simulated_annealing_all():
    sphere = FunctionVisualizer("sphere", 2, -5.12, 5.12)
    sphere.simulated_annealing(sigma=0.5)
    ackley = FunctionVisualizer("ackley", 2, -32.768, 32.768)
    ackley.simulated_annealing(sigma=1.0)
    rastrigin = FunctionVisualizer("rastrigin", 2, -5.12, 5.12)
    rastrigin.simulated_annealing(sigma=0.5)
    rosenbrock = FunctionVisualizer("rosenbrock", 2, -2.048, 2.048)
    rosenbrock.simulated_annealing(sigma=0.1)
    # zoom, normal -600, 600
    griewank = FunctionVisualizer("griewank", 2, -10, 10)
    griewank.simulated_annealing()
    schwefel = FunctionVisualizer("schwefel", 2, -500, 500)
    schwefel.simulated_annealing(sigma=20.0)
    levy = FunctionVisualizer("levy", 2, -10, 10)
    levy.simulated_annealing()
    michalewicz = FunctionVisualizer("michalewicz", 2, 0, np.pi)
    michalewicz.simulated_annealing(sigma=0.1)
    zakharov = FunctionVisualizer("zakharov", 2, -10, 10)
    zakharov.simulated_annealing()


# ALGORITHMS MASS CALL


if __name__ == "__main__":
    print("Start")
    simulated_annealing_all()
