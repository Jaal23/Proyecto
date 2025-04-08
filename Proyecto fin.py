import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

# ======================
# PARÁMETROS AJUSTABLES
# ======================
num_variables = 5
pop_size = 500  # Aumentado para mejor diversidad
mutation_rate = 0.30
num_generations = 1000
bit_length = 16  # Mayor precisión
time_periods = 72

# Límites biológicos
bounds = [
    (50, 500),      # luz
    (100, 2000),    # CO₂
    (6.5, 9.0),     # pH
    (20, 40),       # Temperatura
    (1e5, 1e7)      # Densidad
]

# ======================
# FUNCIONES DE CONVERSIÓN MEJORADAS
# ======================
def real_to_binary(real_values):
    binary_vector = []
    for i, (min_val, max_val) in enumerate(bounds):
        value = real_values[i]
        if i == 4:  # Escala logarítmica para densidad
            log_val = np.log10(value)
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            scaled = (log_val - log_min) / (log_max - log_min)
        else:
            scaled = (value - min_val) / (max_val - min_val)

        int_val = int(scaled * (2**bit_length - 1))
        bits = format(int_val, f'0{bit_length}b')
        binary_vector.extend([int(bit) for bit in bits])
    return np.array(binary_vector)

def binary_to_real(binary_vector):
    real_values = []
    for i in range(num_variables):
        start = i * bit_length
        end = (i+1) * bit_length
        bits = binary_vector[start:end]
        int_val = int("".join(map(str, bits)), 2)

        if i == 4:  # Densidad
            log_min = np.log10(bounds[i][0])
            log_max = np.log10(bounds[i][1])
            log_val = log_min + (int_val / (2**bit_length-1)) * (log_max - log_min)
            real_val = 10**log_val
        else:
            min_val, max_val = bounds[i]
            real_val = min_val + (int_val / (2**bit_length-1)) * (max_val - min_val)

        real_values.append(real_val)
    return np.array(real_values)

# ======================
# FUNCIONES DEL ALGORITMO GENÉTICO MEJORADO
# ======================
def random_solution():
    return np.random.randint(0, 2, size=num_variables*bit_length)

# Función simplificada para la corriente eléctrica I(t)
def calculate_I(t, real_values):
    luz, co2, ph, temp, densidad = real_values
    # Modelo lineal simple (sin dependencia en t)
    return 0.5 * luz + 0.3 * co2 + 0.2 * ph + 0.1 * temp + 0.4 * np.log10(densidad)

def fitness_function(binary_vector):
    real_values = binary_to_real(binary_vector)
    try:
        # Al ser I(t) constante, el promedio en las últimas 24 unidades será el mismo
        I_values = [calculate_I(t, real_values) for t in range(1, time_periods+1)]
        avg_current = np.mean(I_values[-24:])  # Últimas 24 horas
        penalty = 0

        # Penalizaciones por límites
        for i, val in enumerate(real_values):
            if not (bounds[i][0] <= val <= bounds[i][1]):
                penalty += 1000 * abs(val - np.clip(val, bounds[i][0], bounds[i][1]))

        return (real_values[0] * avg_current) / (real_values[1] * real_values[4]) - penalty
    except:
        return -np.inf

def tournament_selection(population, fitness_scores, tournament_size=4):
    indices = np.random.choice(len(population), tournament_size, replace=False)
    contestants = [(fitness_scores[i], i) for i in indices]
    contestants.sort(reverse=True)
    return population[contestants[0][1]].copy()

def crossover(parent1, parent2, method='two_point'):
    if method == 'two_point':
        points = np.random.choice(len(parent1)-1, 2, replace=False)
        points.sort()
        offspring = np.concatenate([
            parent1[:points[0]],
            parent2[points[0]:points[1]],
            parent1[points[1]:]
        ])
    else:  # Uniform crossover
        mask = np.random.randint(0, 2, size=len(parent1))
        offspring = np.where(mask, parent1, parent2)
    return offspring

def mutate(individual, mutation_rate):
    mutation_points = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_points] = 1 - individual[mutation_points]
    return individual

# ======================
# ALGORITMO GENÉTICO COMPLETO CON MECANISMOS AVANZADOS
# ======================
historical_top = defaultdict(int)
adaptive_params = {
    'top_percent': 0.3,
    'mutation_rate': mutation_rate,
    'diversity_threshold': 0.1
}

best_global = {
    'fitness': -np.inf,
    'individual': None,
    'values': None,
    'generation': None
}

population = [random_solution() for _ in range(pop_size)]
avg_fitness_history = []
best_fitness_history = []
population_records = []
diversity_history = []

for gen in range(num_generations):
    # 1. Cálculo de fitness
    fitness_scores = [fitness_function(ind) for ind in population]

    # 2. Actualización del mejor global
    current_best_idx = np.argmax(fitness_scores)
    current_fitness = fitness_scores[current_best_idx]

    if current_fitness > best_global['fitness']:
        best_global.update({
            'fitness': current_fitness,
            'individual': population[current_best_idx].copy(),
            'values': binary_to_real(population[current_best_idx]),
            'generation': gen
        })

    # 3. Parámetros adaptativos
    adaptive_params['mutation_rate'] = mutation_rate * (1.5 - gen/num_generations)
    adaptive_params['top_percent'] = 0.3 - 0.15 * (gen/num_generations)

    # 4. Ordenamiento y selección de élites
    sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    n_top = max(2, int(adaptive_params['top_percent'] * pop_size))
    top_pop = [ind.copy() for ind, _ in sorted_pop[:n_top]]
    top_fit = [fit for _, fit in sorted_pop[:n_top]]

    # 5. Registro histórico
    for ind, _ in sorted_pop[:max(1, int(0.1*pop_size))]:
        historical_top[tuple(ind)] += 1

    # 6. Control de diversidad
    unique_solutions = len(set([tuple(ind) for ind in population]))
    diversity_history.append(unique_solutions/pop_size)

    # 7. Creación de nueva población
    new_pop = []

    # 7.1. Elitismo (10% + mejor histórico)
    new_pop.append(best_global['individual'].copy())
    new_pop.extend(top_pop[:max(1, int(0.1*pop_size)-1)])

    # 7.2. Preservación histórica
    persistent = [np.array(k) for k,v in historical_top.items() if v >= 2]
    new_pop.extend(persistent[:2])

    # 7.3. Generación de descendencia
    while len(new_pop) < pop_size:
        # Selección adaptativa
        if np.random.rand() < 0.7 or gen < num_generations//2:
            p1 = tournament_selection(top_pop, top_fit)
            p2 = tournament_selection(top_pop, top_fit)
        else:
            p1 = tournament_selection(population, fitness_scores)
            p2 = tournament_selection(population, fitness_scores)

        # Crossover probabilístico
        if np.random.rand() < 0.85:
            offspring = crossover(p1, p2)
        else:
            offspring = p1.copy()

        # Mutación adaptativa
        mut_rate = adaptive_params['mutation_rate'] * (1 - unique_solutions/pop_size)
        new_pop.append(mutate(offspring, mut_rate))

    # 8. Aseguramiento del tamaño y actualización
    population = np.array(new_pop[:pop_size])

    # 9. Registros para visualización
    avg_fitness_history.append(np.mean(fitness_scores))
    best_fitness_history.append(best_global['fitness'])
    population_records.append([binary_to_real(ind) for ind in population])

# ======================
# VISUALIZACIONES COMPLETAS
# ======================
# 1. Evolución del fitness
plt.figure(figsize=(12, 6))
plt.plot(avg_fitness_history, label='Fitness Promedio')
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del Fitness")
plt.legend()
plt.grid(True)
plt.show()

# 2. Evolución de parámetros
variables = ['Luz (μmol/m²/s)', 'CO₂ (ppm)', 'pH', 'Temp (°C)', 'Densidad (cél/mL)']
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
axs = axs.flatten()

for i in range(5):
    gen_values = [np.mean([ind[i] for ind in gen]) for gen in population_records]
    axs[i].plot(gen_values, color='tab:blue')
    axs[i].set_title(variables[i])
    axs[i].grid(True)

# 3. Diversidad poblacional
axs[5].plot(diversity_history, color='tab:purple')
axs[5].set_title('Diversidad Poblacional')
axs[5].set_ylabel('% Soluciones Únicas')
axs[5].grid(True)
plt.tight_layout()
plt.show()

# 4. Matriz de correlación final
df_final = pd.DataFrame(population_records[-1], columns=variables)
plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlación entre Variables (Última Generación)")
plt.show()


# ======================
# RESULTADOS FINALES
# ======================
print("=== MEJOR SOLUCIÓN ENCONTRADA ===")
print(f"Generación: {best_global['generation']}")
print(f"Fitness: {best_global['fitness']:.4f}\n")

params = best_global['values']
print("Parámetros Óptimos:")
print(f"• Intensidad Lumínica: {params[0]:.1f} μmol/m²/s")
print(f"• Concentración CO₂: {params[1]:.0f} ppm")
print(f"• pH: {params[2]:.2f}")
print(f"• Temperatura: {params[3]:.1f}°C")
print(f"• Densidad Algal: {params[4]:.2e} células/mL\n")

print("=== ANÁLISIS DE CONVERGENCIA ===")
print(f"Diversidad Final: {diversity_history[-1]*100:.1f}% soluciones únicas")
print(f"Mejora Total: {(best_fitness_history[-1] - best_fitness_history[0])/best_fitness_history[0]*100:.1f}%")
