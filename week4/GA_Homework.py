# Genetic Algorithm for solving a Diophantine equation: a + 2b + 3c + 4d = 30
# Each chromosome is a tuple (a, b, c, d) with 1 <= a,b,c,d <= 30

import random

# Problem and GA parameters
TARGET_SUM = 30  # Target value for the equation
POPULATION_SIZE = 5  # Number of chromosomes in population
GENES_PER_CHROMOSOME = 4  # Number of genes per chromosome
GENE_MIN = 1  # Minimum gene value
GENE_MAX = 30  # Maximum gene value
MUTATION_RATE = 0.1  # Probability of mutation per chromosome


# Fitness function: lower is better, 0 means solution found
def calculate_fitness(chromosome):
    a, b, c, d = chromosome
    return abs((a + 2 * b + 3 * c + 4 * d) - TARGET_SUM)


# Generate initial population of random chromosomes
def generate_initial_population():
    return [
        tuple(random.randint(GENE_MIN, GENE_MAX) for _ in range(GENES_PER_CHROMOSOME))
        for _ in range(POPULATION_SIZE)
    ]


# Roulette wheel selection: probabilistically select a parent based on fitness rates
def roulette_wheel_selection(population, fitness_rates):
    pick = random.uniform(0, sum(fitness_rates))
    current = 0
    for chrom, rate in zip(population, fitness_rates):
        current += rate
        if current > pick:
            return chrom
    return population[-1]


# Single-point crossover: combine genes from two parents
def crossover(parent1, parent2):
    point = random.randint(1, GENES_PER_CHROMOSOME - 1)
    print(f"\nCrossover point = {point}")
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


# Mutate a chromosome: randomly change one gene with probability MUTATION_RATE
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, GENES_PER_CHROMOSOME - 1)
        new_val = random.randint(GENE_MIN, GENE_MAX)
        chromosome = list(chromosome)
        chromosome[index] = new_val
        return tuple(chromosome)
    return chromosome


# Force mutation: always change one gene
def force_mutate(chromosome):
    index = random.randint(0, GENES_PER_CHROMOSOME - 1)
    new_val = random.randint(GENE_MIN, GENE_MAX)
    chromosome = list(chromosome)
    chromosome[index] = new_val
    return tuple(chromosome)


# Main genetic algorithm loop
def genetic_algorithm():
    generation = 1
    population = generate_initial_population()

    prev_population = None
    prev_fitness_rates = None

    while True:
        # Report the chromosome with the lowest fitness rate from previous population
        if prev_population is not None and prev_fitness_rates is not None:
            min_rate = min(prev_fitness_rates)
            min_index = prev_fitness_rates.index(min_rate)
            print(
                f"\nChromosome c{min_index+1}: {prev_population[min_index]} is invalid for cross-over (fitnessRate = {min_rate:.2f}%)"
            )

        # Print current population and fitness info
        print(f"\nCrossOver = {generation}")
        print(
            f"{'Chromosome No.':<17}{'Chromosome':<23}{'Fitness':<12}{'FitnessRate(%)'}"
        )

        fitnesses = [calculate_fitness(ch) for ch in population]
        total_inverse_fitness = sum([1 / (1 + f) for f in fitnesses])

        fitness_rates = []
        solution_index = None
        for i, (chrom, fit) in enumerate(zip(population, fitnesses)):
            if fit == 0 and solution_index is None:
                solution_index = i

        for i, (chrom, fit) in enumerate(zip(population, fitnesses)):
            if solution_index is not None:
                rate = 100.0 if i == solution_index else 0.0
            else:
                rate = (1 / (1 + fit)) / total_inverse_fitness * 100
            fitness_rates.append(rate)
            print(f"C{i+1:<15}{str(chrom):<25}{fit:<12}{rate:.2f}")

        # Save current population/rates for next loop
        prev_population = population.copy()
        prev_fitness_rates = fitness_rates.copy()

        # Stopping condition: solution found
        if solution_index is not None:
            print(
                f"\n>> ✅ Solution found in Chromosome C{solution_index+1}: {population[solution_index]}"
            )
            print(
                "No more Crossover is needed due to the best fitness value (0) and fitnessRate(100%)"
            )
            break

        # Stagnation: all chromosomes have same fitness
        if len(set(fitnesses)) == 1:
            print(
                "\n⚠️ All chromosomes have the same fitness. Applying mutation to escape local stagnation..."
            )
            population = [force_mutate(ch) for ch in population]
            generation += 1
            continue

        # Parent selection for next generation
        parent_pairs = []
        for _ in range(POPULATION_SIZE):
            parent1 = roulette_wheel_selection(population, fitness_rates)
            parent2 = roulette_wheel_selection(population, fitness_rates)
            parent_pairs.append((parent1, parent2))

        # Generate new population via crossover and mutation
        new_population = []
        for parent1, parent2 in parent_pairs:
            child1, _ = crossover(parent1, parent2)
            child1 = mutate(child1)
            # If child is a solution, force mutation to avoid premature convergence
            if calculate_fitness(child1) == 0:
                print(
                    f"\nChromosome {child1} is invalid for cross-over (fitness = 0). Mutating..."
                )
                child1 = mutate(child1)
                print(f"-> Mutated to: {child1}")
            new_population.append(child1)
        population = new_population[:POPULATION_SIZE]
        generation += 1

        # User interaction for next step
        try:
            cont = input(
                "\nEnter 1 for next Cross-over/Fitness values, 2 to restart Initial Population, or any Integer to Exit: "
            )
        except EOFError:
            print("Exiting program.")
            break

        if cont == "2":
            population = generate_initial_population()
            generation = 1
        elif cont != "1":
            print("Exiting program.")
            break


if __name__ == "__main__":
    genetic_algorithm()
