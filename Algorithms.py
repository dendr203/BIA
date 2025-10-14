import os
import numpy as np
import random

import plotly.graph_objects as go

resolution = 500
graph_folder_name = "graphs_files"


class FunctionVisualizer:
    def __init__(
        self, func_name, dimensions, lower_bound, upper_bound, make_new_file=False):
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

    def visualise(self, X, Y, Z, label, highlight_points=None, history_temp=None, best_point=None, migration_paths=None):
        fig = go.Figure()
        
        z_min = float(np.nanmin(Z))
        z_max = float(np.nanmax(Z))

        # Přidání povrchu (surface)
        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, opacity=1, colorscale="jet", showscale=False)
        )

        # Přidání scatter bodů, pokud je předám
        if highlight_points is not None:
            x_vals = highlight_points[0]
            y_vals = highlight_points[1]
            z_vals = highlight_points[2]
            
            z_min = min(z_min, np.min(z_vals))
            z_max = max(z_max, np.max(z_vals))
            
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
            print(f"Best point: x={bx}, y={by}, f={bf}")
            fig.add_trace(go.Scatter3d(
                x=[bx], y=[by], z=[bf],
                mode="markers",
                marker=dict(
                    size=10,
                    color="lime",
                    symbol="diamond",
                    line=dict(
                        color="black",
                        width=2)
                    ),
                name="Best point",
                text=["BEST"],
                textposition="top center",
                hoverinfo="text+x+y+z"
        ))
        
        if best_point is not None:
        # drobné rozšíření rozsahu o margin
            margin = 0.05 * max(abs(z_max - z_min), 1e-6)
            z_min = min(z_min, bf) - margin
            z_max = max(z_max, bf) + margin
        
        if migration_paths is not None:
            for path in migration_paths:
                fig.add_trace(go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=[self.function_type(p) for p in path],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    name='Migration Path'
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
                zaxis=dict(range=[z_min, z_max]),
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

    def differencial_evolution(self, pop_size=15, generations=20, F=0.8, CR=0.8):
        #F (mutation faktor) - větší = agresivnější průzkum
        #CR (crossover rate) - pravděpodobnost křížení, tedy větší = více prvků beru z mutanta, menší je konzervativnější
        
        X, Y, Z = self.compute()
        
        # inicializace populace (pop_size x d) náhodných vektorů v d dimenzích
        pop = np.random.uniform(self.lB, self.uB, size=(pop_size, self.d))
        fitness = np.array([self.function_type(ind) for ind in pop])

        # počáteční bod náhodně vybraný z předem vypočítaných pro lepší vizualizaci algoritmů
        # best_idx = np.argmin(fitness)
        best_idx = random.randint(0, len(fitness))
        best = pop[best_idx].copy()
        best_val = fitness[best_idx]
        
        # historie nejlepších pro vizualizaci
        best_x_history = [best[0]]
        best_y_history = [best[1]]
        best_f_history = [best_val]
        
        for gen in range(generations):
            for i in range(pop_size):
                # výběr tří různé index kromě i
                idxs = list(range(pop_size))
                idxs = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                
                xr1 = pop[r1]
                xr2 = pop[r2]
                xr3 = pop[r3]
                
                # vytvoření mutanta (DE/rand/1/bin) + clip do hranic => výsledek, bod mezi těmito třemi jedinci
                mutant = xr1 + F * (xr2 - xr3)
                mutant = np.clip(mutant, self.lB, self.uB)

                # binomální křížení s cílem vytvořit trial vector + zajištění, že alespoň jeden prvek pochází z mutanta
                cross_points = np.random.rand(self.d) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.d)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # selekce: pokud je trial lepší než cílový jedinec, nahradí ho
                trial_fitness = self.function_type(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    
                    # aktualizace nejlepšího řešení
                    if trial_fitness < best_val:
                        best = trial.copy()
                        best_val = trial_fitness

            best_x_history.append(best[0])
            best_y_history.append(best[1])
            best_f_history.append(best_val)
        
        print(f"Differential Evolution result: {best}, f_value: {best_val}")
        history_coord = np.array([best_x_history, best_y_history, best_f_history])
        self.visualise(X, Y, Z, "Differential Evolution", highlight_points=history_coord, best_point=(best[0], best[1], best_val))
        return

    def particle_swarm_optimization(self, swarm_size=15, migrations=100, w=0.7, c1=1.5, c2=1.0):
        #W (setrvačnost) - větší = lepší pokrytí prostoru
        #c1 (vliv vlastní paměti) - vysoké = částice se drží své pozice
        #c2 (vliv globálního minima) - vysoké = částice se stahují ke globálnímu řešení
        
        X, Y, Z = self.compute()
        
        #inicializace hejna
        positions = np.random.uniform(self.lB, self.uB, size=(swarm_size, self.d))
        velocities = np.zeros_like(positions)
        fitness = np.array([self.function_type(ind) for ind in positions])
        
        
        # paměť nejlepší pozice každé částice v hejnu
        pbest_positions = positions.copy()
        pbest_values = fitness.copy()
        
        # globální nejlepší - v 0 iteraci se vybere náhodný nejlepší z předpočítených (lepší pro vizualizaci algoritmu)
        #gbest_idx = np.argmin(fitness)
        gbest_idx = random.randint(0, len(fitness))
        gbest_position = positions[gbest_idx].copy()
        gbest_value = fitness[gbest_idx]
        
        # historie nejlepších pro vizualizaci
        best_x_history = [gbest_position[0]]
        best_y_history = [gbest_position[1]]
        best_f_history = [gbest_value]
        
        
        for gen in range(migrations):
            for i in range(swarm_size):
                r1 = np.random.rand(self.d)
                r2 = np.random.rand(self.d)
                
                # aktualizace rychlosti
                velocities[i] = (
                    w * velocities[i]                                   #setrvačnost - částečně zachová směr pohybu
                    + c1 * r1 * (pbest_positions[i] - positions[i])     #kognitivní složka - táhne částice směrem k jejímu nejlepšímu nalezenému řešení
                    + c2 * r2 * (gbest_position - positions[i])         #sociální složka - táhne částice směrem k (zatím) nejlepšímu řešení hejna
                    )
                
                # aktualizace pozice + clip
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lB, self.uB)
                
                current_fitness = self.function_type(positions[i])
                if current_fitness < pbest_values[i]:
                    pbest_positions[i] = positions[i].copy()
                    pbest_values[i] = current_fitness
                    
                    if current_fitness < gbest_value:
                        gbest_position = positions[i].copy()
                        gbest_value = current_fitness
            
            best_x_history.append(gbest_position[0])
            best_y_history.append(gbest_position[1])
            best_f_history.append(gbest_value)
        
        print(f"PSO result: {gbest_position}, f_value: {gbest_value}")
        history_coord = np.array([best_x_history, best_y_history, best_f_history])
        self.visualise(X, Y, Z, "Particle Swarm Optimization", highlight_points=history_coord, best_point=(gbest_position[0], gbest_position[1], gbest_value))
        return
    
    def SOMA_allToOne(self, pop_size=20, migrations=100, path_lenght=3.0, step=0.11, perturbation=0.4):
        #path_lenght - délka přímky k vůdci hejna
        #step - krok mezi body na trajektorii
        #perturbation - náhodná odchylka v každé dimenzi přičená ke každému bodu na cestě 
        
        X, Y, Z = self.compute()
        
        #inicializace hejna
        pop = np.random.uniform(self.lB, self.uB, size=(pop_size, self.d))
        fitness = np.array([self.function_type(ind) for ind in pop])
        
        # historie nejlepších pro vizualizaci
        # best_idx = np.argmin(fitness)
        best_idx = random.randint(0, len(fitness))
        leader = pop[best_idx].copy()
        leader_val = fitness[best_idx]
        
        best_x_history = [leader[0]]
        best_y_history = [leader[1]]
        best_f_history = [leader_val]
        
        migration_paths = []
        
        for gen in range(migrations):
            for i in range(pop_size):
                if i == best_idx:
                    continue # vůdce nemigruje
                
                path = []
                t = 0.0
                candidate = pop[i].copy()
                best_candidate = candidate.copy()
                best_candidate_val = self.function_type(candidate)
                
                while t <= path_lenght:
                    # migrace směrem k vůdci
                    step_vector = candidate + (leader - candidate) * t
                    # perturbace - aby se nešlo přesně po přímce a mohli jsme najít lepší minima
                    step_vector += np.random.uniform(-perturbation, perturbation, self.d)
                    step_vector = np.clip(step_vector, self.lB, self.uB)
                    path.append(step_vector.copy())
                    
                    val = self.function_type(step_vector)
                    if val < best_candidate_val:
                        best_candidate = step_vector.copy()
                        best_candidate_val = val
                    
                    t += step
                
                # aktualizace jedince
                pop[i] = best_candidate
                fitness[i] = best_candidate_val
                migration_paths.append(np.array(path))
            
            # aktualizace vůdce
            best_idx = np.argmin(fitness)
            leader = pop[best_idx].copy()
            leader_val = fitness[best_idx]
            
            best_x_history.append(leader[0])
            best_y_history.append(leader[1])
            best_f_history.append(leader_val)
        
        print(f"SOMA result: {leader}, f_value: {leader_val}")
        history_coord = np.array([best_x_history, best_y_history, best_f_history])
        self.visualise(X, Y, Z, "Particle Swarm Optimization", highlight_points=history_coord, best_point=(leader[0], leader[1], leader_val))
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

def differencial_eveolution_all():
    sphere = FunctionVisualizer("sphere", 2, -5.12, 5.12)
    sphere.differencial_evolution()
    ackley = FunctionVisualizer("ackley", 2, -32.768, 32.768)
    ackley.differencial_evolution()
    rastrigin = FunctionVisualizer("rastrigin", 2, -5.12, 5.12)
    rastrigin.differencial_evolution()
    rosenbrock = FunctionVisualizer("rosenbrock", 2, -2.048, 2.048)
    rosenbrock.differencial_evolution()
    # zoom, normal -600, 600
    griewank = FunctionVisualizer("griewank", 2, -10, 10)
    griewank.differencial_evolution()
    schwefel = FunctionVisualizer("schwefel", 2, -500, 500)
    schwefel.differencial_evolution()
    levy = FunctionVisualizer("levy", 2, -10, 10)
    levy.differencial_evolution()
    michalewicz = FunctionVisualizer("michalewicz", 2, 0, np.pi)
    michalewicz.differencial_evolution()
    zakharov = FunctionVisualizer("zakharov", 2, -10, 10)
    zakharov.differencial_evolution()

def particle_swarm_optimization_all():
    sphere = FunctionVisualizer("sphere", 2, -5.12, 5.12)
    sphere.particle_swarm_optimization(w=0.5, c1=1.5, c2=1.5)
    ackley = FunctionVisualizer("ackley", 2, -32.768, 32.768)
    ackley.particle_swarm_optimization()
    rastrigin = FunctionVisualizer("rastrigin", 2, -5.12, 5.12)
    rastrigin.particle_swarm_optimization(w=0.7, c1=2.0, c2=2.0)
    rosenbrock = FunctionVisualizer("rosenbrock", 2, -2.048, 2.048)
    rosenbrock.particle_swarm_optimization()
    # zoom, normal -600, 600
    griewank = FunctionVisualizer("griewank", 2, -10, 10)
    griewank.particle_swarm_optimization(w=0.7, c1=2.0, c2=2.0)
    schwefel = FunctionVisualizer("schwefel", 2, -500, 500)
    schwefel.particle_swarm_optimization(w=0.7, c1=2.0, c2=2.0)
    levy = FunctionVisualizer("levy", 2, -10, 10)
    levy.particle_swarm_optimization()
    michalewicz = FunctionVisualizer("michalewicz", 2, 0, np.pi)
    michalewicz.particle_swarm_optimization()
    zakharov = FunctionVisualizer("zakharov", 2, -10, 10)
    zakharov.particle_swarm_optimization()

def SOMA_allToOne_all():
    sphere = FunctionVisualizer("sphere", 2, -5.12, 5.12)
    sphere.SOMA_allToOne()
    ackley = FunctionVisualizer("ackley", 2, -32.768, 32.768)
    ackley.SOMA_allToOne()
    rastrigin = FunctionVisualizer("rastrigin", 2, -5.12, 5.12)
    rastrigin.SOMA_allToOne()
    rosenbrock = FunctionVisualizer("rosenbrock", 2, -2.048, 2.048)
    rosenbrock.SOMA_allToOne()
    # zoom, normal -600, 600
    griewank = FunctionVisualizer("griewank", 2, -10, 10)
    griewank.SOMA_allToOne()
    schwefel = FunctionVisualizer("schwefel", 2, -500, 500)
    schwefel.SOMA_allToOne()
    levy = FunctionVisualizer("levy", 2, -10, 10)
    levy.SOMA_allToOne()
    michalewicz = FunctionVisualizer("michalewicz", 2, 0, np.pi)
    michalewicz.SOMA_allToOne()
    zakharov = FunctionVisualizer("zakharov", 2, -10, 10)
    zakharov.SOMA_allToOne()

# ALGORITHMS MASS CALL


if __name__ == "__main__":
    print("Start")
    SOMA_allToOne_all()
