import time
import random
from hashcode_solvers import ScoreSquareSolver, SimpleScoreVarianceSolver, SquareScoreVarianceSolver, BookCountPowerSolver, \
    RandomSolver, PowerSolver, BookCountSolver, BookNumbersSolver, HeuristicSolver, ProblemSolver, GreedyIntervalSolver, \
    MutationHillClimbingSolver
import copy

# Max flow Max cost solver
from ortools.graph import pywrapgraph

start_time = time.time()

solver_methods = [HeuristicSolver, PowerSolver, SimpleScoreVarianceSolver, SquareScoreVarianceSolver,
                  BookNumbersSolver, ScoreSquareSolver, BookCountSolver, BookCountPowerSolver, RandomSolver]


class Library:
    def __init__(self, index, N, T, M):
        self.id = index
        self.size = N
        self.signup_time = T
        self.books_per_day = M
        self.book_ids = set()

    def add_book(self, book):
        self.book_ids.add(book)

    def remove_book(self, book):
        self.book_ids.remove(book)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# Loads data from files into the class Library Objects
def process_file():
    B, L, D = input().split()
    B, L, D = int(B), int(L), int(D)

    book_libraries = [set() for i in range(0, B)]
    book_values = [int(n) for n in input().split()]
    libraries = []

    for i in range(L):
        N, T, M = input().split()
        N, T, M = int(N), int(T), int(M)
        book_ids = set(int(id_) for id_ in input().split())
        library = Library(i, N, T, M)
        
        for book_id in book_ids:
            book_libraries[book_id].add(i)
            library.add_book(book_id)
        libraries.append(library)
    return (B, L, D), book_values, book_libraries, libraries


# Checks if provided solution is correct
def check_solution(D, libraries):
    days = 0
    prev_books = set()
    lib_ids = [lib.id for lib in libraries]
    # checking if there are no duplicates of libraries in the solution
    assert (len(lib_ids) == len(set(lib_ids)))

    for library in libraries:
        days += library.signup_time

        # checking if library sends correct number of books
        if len(library.book_ids) > (D - days) * library.books_per_day:
            print("Library sends more books, than it is allowed to:", len(library.book_ids),
                  (D - days) * library.books_per_day)

        # checking if there is no repetition of books
        assert (len(library.book_ids) == len(set(library.book_ids)))
        assert (not any([(book in prev_books) for book in library.book_ids]))
        prev_books.update(library.book_ids)

    # checking if we do not go over the deadline
    assert (days < D)


# Returns score of provided solution
def score_solution(libraries, book_values):
    score = 0
    for library in libraries:
        for book_id in library.book_ids:
            score += book_values[book_id]
    return score


class GeneticSolver(ProblemSolver):
    def __init__(self, B, L, D, book_values, book_libraries, libraries,start_time, deadline = 240,pop_size=20, p_mutate=0.5, surv_rate=0.2,
                 tournament_size=3):
        super().__init__(B, L, D, book_values, book_libraries, libraries)
        num = 1
        length_of_libraries = len(libraries)
        while num < pop_size and length_of_libraries != 1:
            num *= length_of_libraries
            length_of_libraries -= 1
        self.pop_size = min(pop_size, num)
        self.p_mutate = p_mutate
        self.tournament_size = min(self.pop_size, tournament_size)
        self.survival_rate = surv_rate
        self.book_values = book_values
        self.individual_scores = dict()
        self.lib_ids = [i for i in range(len(libraries))]
        self.lib_scores = [self.lib_score(lib) for lib in self.libraries]
        self.start_time = start_time
        self.deadline = deadline

    def lib_score(self, lib):
        delta_time = self.D - lib.signup_time
        n_best_books = self.get_n_best_books(lib, delta_time * lib.books_per_day)
        sum_of_best_book_scores = sum([self.book_values[book] for book in n_best_books])
        return sum_of_best_book_scores / lib.signup_time

    def mutate(self, individual):
        new_one = list(individual)
        if random.random() <= 0.5:
            # internal mutation
            length = len(new_one)
            for i in range(4):
                a, b = random.sample(range(length), 2)
                new_one[a], new_one[b] = new_one[b], new_one[a]
            return tuple(individual)
        else:
            # external mutation
            additional = random.choices(self.lib_ids, k=4, weights=self.lib_scores)
            unique = set(additional)
            for un in unique:
                new_one.insert(random.randrange(0, len(new_one)), un)
            return self.cut(new_one)

    def cut(self, new_one):
        day = 0
        chosen = set()
        solution = []
        for i in range(len(new_one)):
            if new_one[i] not in chosen:
                day += self.libraries[new_one[i]].signup_time
                if day >= self.D:
                    continue
                solution.append(new_one[i])
                chosen.add(new_one[i])
        return tuple(solution)

    def crossover(self, individual_1, individual_2):
        set1 = set(individual_1)
        set2 = set(individual_2)
        common = set1.intersection(set2)
        max_length = max(len(individual_1), len(individual_2))
        child1, child2, = [-1] * max_length, [-1] * max_length
        only1, only2 = [], []
        for i in range(len(individual_1)):
            if individual_1[i] in common:
                child1[i] = individual_1[i]
            else:
                only1.append(individual_1[i])
        for i in range(len(individual_2)):
            if individual_2[i] in common:
                child2[i] = individual_2[i]
            else:
                only2.append(individual_2[i])
        for i in range(max_length):
            if child1[i] == -1 and len(only2) > 0:
                child1[i] = only2.pop(0)
            if child2[i] == -1 and len(only1) > 0:
                child2[i] = only1.pop(0)
        child1 = [i for i in child1 if i != -1]
        child2 = [i for i in child2 if i != -1]

        child1 = self.cut(child1)
        child2 = self.cut(child2)
        if random.random() <= self.p_mutate:
            child1 = self.mutate(child1)
        if random.random() <= self.p_mutate:
            child2 = self.mutate(child2)
        return child1, child2

    def tournament(self, indivs):
        return max(indivs, key=self.individual_scores.get)

    def get_initial_population(self):
        solvers_cls = [HeuristicSolver, PowerSolver, SimpleScoreVarianceSolver, SquareScoreVarianceSolver,
                       BookNumbersSolver, ScoreSquareSolver, BookCountSolver, BookCountPowerSolver]
        solvers = [cl(self.B, self.L, self.D, self.book_values, self.book_libraries, self.libraries) for cl in
                   solvers_cls]
        population = [solver.get_individual() for solver in solvers]
        greedy_interval_solver = GreedyIntervalSolver(self.B, self.L, self.D, self.book_values, self.book_libraries,
                                                      self.libraries)
        greedy_interval_solution = greedy_interval_solver.get_solution()
        gis_individual = greedy_interval_solver.get_individual_from_solution(greedy_interval_solution)
        population.append(gis_individual)
        self.individual_scores[gis_individual] = score_solution(greedy_interval_solution, self.book_values)
        random_solver = RandomSolver(self.B, self.L, self.D, self.book_values, self.book_libraries, self.libraries)
        while len(population) < self.pop_size:
            population.append(random_solver.get_individual())
        return population

    def get_individual(self):
        population = self.get_initial_population()
        for a_individual in population:
            sol = self.get_solution(a_individual)
            self.individual_scores[a_individual] = score_solution(sol, self.book_values)

        while True:
            if time.time() - self.start_time > self.deadline:
                break
            new_population = [self.tournament(random.sample(population, self.tournament_size)) for i in
                              range(int(self.pop_size * self.survival_rate))]
            while len(new_population) < self.pop_size:
                individual_1 = self.tournament(random.sample(population, self.tournament_size))
                individual_2 = self.tournament(random.sample(population, self.tournament_size))
                if random.random() <= 0.5:
                    new_child1 = self.mutate(individual_1)
                    new_child2 = self.mutate(individual_2)
                else:
                    new_child1, new_child2 = self.crossover(individual_1, individual_2)
                if new_child1 not in self.individual_scores:
                    self.individual_scores[new_child1] = score_solution(self.get_solution(new_child1), self.book_values)
                if new_child2 not in self.individual_scores:
                    self.individual_scores[new_child2] = score_solution(self.get_solution(new_child2), self.book_values)
                new_population.append(new_child1)
                new_population.append(new_child2)
            population = tuple(new_population)
        return max(self.individual_scores, key=self.individual_scores.get)


def generate_book_to_node_assignment(number_of_libraries, all_book_ids):
    book_to_node = dict()
    node_to_book = dict()
    counter = 1
    for book_id in all_book_ids:
        node_nr = number_of_libraries + counter
        book_to_node[book_id] = node_nr
        node_to_book[node_nr] = book_id
        counter += 1
    return book_to_node, node_to_book


def lib_to_node(lib_id):
    return lib_id + 1


def node_to_lib(node):
    return node - 1


def create_solution(solution):
    lines = [str(len(solution))]
    for library in solution:
        lines.append(f"{str(library.id)} {len(library.book_ids)}")
        a_string = ""
        for book_id in library.book_ids:
            a_string += f"{str(book_id)} "
        a_string = a_string.strip()
        lines.append(a_string)
    for line in lines:
        print(line)


def solve_one_example_with_genetic_climber_max_flow(deadline_for_genetic, deadline_for_hill_climber):
    (B, L, D), book_values, book_counts, libraries = process_file()
    libraries_backup = copy.deepcopy(libraries)
    genetic_solver = GeneticSolver(B, L, D, book_values, book_counts, libraries, start_time,
                                   deadline=deadline_for_genetic)
    individual = genetic_solver.get_individual()
    check_solution(D, genetic_solver.get_solution(selected_lib_ids=individual))

    hill_climber_solver = MutationHillClimbingSolver(B, L, D, book_values, book_counts, libraries,
                                                     genetic_solver.individual_scores,start_time,
                                                     deadline=deadline_for_hill_climber)
    # Creating new individual with hill_climber
    individual = hill_climber_solver.get_individual(individual)
    climbed_solution = hill_climber_solver.get_solution(individual)
    check_solution(D, climbed_solution)
    libraries = libraries_backup

    all_book_ids = set()
    for lib_id in individual:
        all_book_ids.update(libraries[lib_id].book_ids)
    book_to_node, node_to_book = generate_book_to_node_assignment(len(individual), all_book_ids)

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []

    source = 0
    sink = len(individual) + len(all_book_ids) + 1
    day = 0

    # from source to lib
    for i in range(len(individual)):
        day += libraries[individual[i]].signup_time
        start_nodes.append(source)
        end_nodes.append(lib_to_node(i))
        capacities.append((D - day) * libraries[individual[i]].books_per_day)
        unit_costs.append(0)

    # from lib to book
    for i in range(len(individual)):
        for book_id in libraries[individual[i]].book_ids:
            start_nodes.append(lib_to_node(i))
            end_nodes.append(book_to_node[book_id])
            capacities.append(1)
            unit_costs.append(0)

    # from book to sink
    for book_id in all_book_ids:
        start_nodes.append(book_to_node[book_id])
        end_nodes.append(sink)
        capacities.append(1)
        unit_costs.append(-book_values[book_id])

    max_flow = pywrapgraph.SimpleMaxFlow()
    # Add each arc.
    for i in range(0, len(start_nodes)):
        max_flow.AddArcWithCapacity(start_nodes[i], end_nodes[i], capacities[i])
    max_flow.Solve(0, sink)

    if max_flow.Solve(0, sink) != max_flow.OPTIMAL:
        raise Exception("Failed with max_flow", max_flow.OptimalFlow())

    optimal_flow = max_flow.OptimalFlow()

    supplies = [0 for i in range(sink + 1)]
    supplies[0] = optimal_flow
    supplies[-1] = -optimal_flow

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], unit_costs[i])
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    if min_cost_flow.Solve() != min_cost_flow.OPTIMAL:
        raise Exception("Failed with min_cost_flow", max_flow.OptimalFlow())

    solution = []
    for lib_id in individual:
        lib2 = copy.deepcopy(libraries[lib_id])
        lib2.book_ids = set()
        solution.append(lib2)
    for i in range(min_cost_flow.NumArcs()):
        from_node = min_cost_flow.Tail(i)
        to_node = min_cost_flow.Head(i)
        flow = min_cost_flow.Flow(i)

        if lib_to_node(0) <= from_node <= lib_to_node(len(individual) - 1) and flow == 1:
            lib_index = node_to_lib(from_node)
            book_id = node_to_book[to_node]
            solution[lib_index].book_ids.add(book_id)

    check_solution(D, solution)
    create_solution(solution)


if __name__ == "__main__":
    solve_one_example_with_genetic_climber_max_flow(260, 280)
