import copy
import numpy as np
from math import sqrt
import random
import time


def score_solution(libraries, book_values):
    score = 0
    for library in libraries:
        for book_id in library.book_ids:
            score += book_values[book_id]
    return score


class ProblemSolver:
    def __init__(self, B, L, D, book_values, book_libraries, libraries):
        self.B, self.L, self.D = B, L, D
        self.book_values = book_values
        self.libraries = libraries
        self.book_libraries = copy.deepcopy(book_libraries)

    def book_score(self, book):
        return self.book_values[book]

    def get_n_best_books(self, lib, n):
        return sorted(lib.book_ids, key=self.book_score, reverse=True)[:n]

    def get_solution(self, selected_lib_ids=None):
        if not selected_lib_ids:
            selected_lib_ids = self.get_individual()
        day = 0
        selected_libraries = [copy.deepcopy(self.libraries[i]) for i in selected_lib_ids]
        it = 0
        already_scanned_books = set()
        while it < len(selected_libraries):
            next_library = selected_libraries[it]
            day += next_library.signup_time
            if day >= self.D:
                break
            next_library.book_ids = next_library.book_ids - already_scanned_books
            next_library.book_ids = sorted(next_library.book_ids, key=self.book_score, reverse=True)[
                                    :(self.D - day) * next_library.books_per_day]
            already_scanned_books.update(next_library.book_ids)
            it += 1
        return selected_libraries[:it]

    def get_individual(self):
        pass


class HeuristicSolver(ProblemSolver):
    def library_score(self, lib_id):
        lib = self.libraries[lib_id]
        delta_time = self.D - lib.signup_time
        n_best_books = self.get_n_best_books(lib, delta_time * lib.books_per_day)
        sum_of_best_book_scores = sum([self.book_values[book] for book in n_best_books])
        return sum_of_best_book_scores / lib.signup_time

    def get_individual(self):
        lib_ids = [i for i in range(len(self.libraries))]
        lib_ids.sort(key=self.library_score, reverse=True)
        day = 0
        chosen = []
        for i in range(len(lib_ids)):
            if day + self.libraries[lib_ids[i]].signup_time >= self.D:
                continue
            day += self.libraries[lib_ids[i]].signup_time
            chosen.append(lib_ids[i])
        return tuple(chosen)


class PowerSolver(HeuristicSolver):
    def library_score(self, lib_id):
        lib = self.libraries[lib_id]
        delta_time = self.D - lib.signup_time
        n_best_books = self.get_n_best_books(lib, delta_time * lib.books_per_day)
        sum_of_best_book_scores = sum([self.book_values[book] for book in n_best_books])
        return sum_of_best_book_scores / lib.signup_time ** (1 + lib.signup_time / self.D)


class SimpleScoreVarianceSolver(HeuristicSolver):
    def library_score(self, lib_id):
        lib = self.libraries[lib_id]
        delta_time = self.D - lib.signup_time
        n_best_books = self.get_n_best_books(lib, delta_time * lib.books_per_day)
        best_scores = [self.book_values[book] for book in n_best_books]
        sum_of_best_book_scores = sum(best_scores)
        book_variance = max(0.001, np.var(best_scores))
        return sum_of_best_book_scores / book_variance


class SquareScoreVarianceSolver(HeuristicSolver):
    def library_score(self, lib_id):
        lib = self.libraries[lib_id]
        delta_time = self.D - lib.signup_time
        n_best_books = self.get_n_best_books(lib, delta_time * lib.books_per_day)
        best_scores = [self.book_values[book] for book in n_best_books]
        sum_of_best_book_scores = sum(best_scores)
        book_variance = max(0.001, np.var(best_scores))
        return sum_of_best_book_scores ** 2 / (lib.signup_time * lib.signup_time * sqrt(book_variance))


class BookNumbersSolver(HeuristicSolver):
    def library_score(self, lib_id):
        lib = self.libraries[lib_id]
        delta_time = self.D - lib.signup_time
        n_best_books = self.get_n_best_books(lib, delta_time * lib.books_per_day)
        best_scores = [self.book_values[book] for book in n_best_books]
        sum_of_best_book_scores = sum(best_scores)
        book_variance = max(0.001, np.var(best_scores))
        return sum_of_best_book_scores ** 2 / (book_variance * len(n_best_books) * lib.signup_time)


class ScoreSquareSolver(HeuristicSolver):
    def library_score(self, lib_id):
        lib = self.libraries[lib_id]
        delta_time = self.D - lib.signup_time
        n_best_books = self.get_n_best_books(lib, delta_time * lib.books_per_day)
        sum_of_best_book_scores = sum([self.book_values[book] for book in n_best_books])
        return sum_of_best_book_scores ** 2 / lib.signup_time


class BookCountSolver(HeuristicSolver):
    def book_score(self, book):
        return self.book_values[book] - 0.7 * len(self.book_libraries[book])


class BookCountPowerSolver(PowerSolver):
    def book_score(self, book):
        return self.book_values[book] - 0.7 * len(self.book_libraries[book])


class RandomSolver(ProblemSolver):
    def get_individual(self):
        lib_ids = [i for i in range(len(self.libraries))]
        random.shuffle(lib_ids)
        day = 0
        for i in range(len(lib_ids)):
            day += self.libraries[lib_ids[i]].signup_time
            if day >= self.D:
                return tuple(lib_ids[:i])
        return tuple(lib_ids)


class GreedyIntervalSolver:
    def __init__(self, B, L, D, book_values, book_libraries, libraries):
        self.B, self.L, self.D = B, L, D
        self.book_values = book_values
        self.libraries = copy.deepcopy(libraries)
        self.book_libraries = copy.deepcopy(book_libraries)

    def book_score(self, book_id):
        return self.book_values[book_id] - 0.7 * len(self.book_libraries[book_id])

    def get_n_best_books(self, lib, n):
        return sorted(lib.book_ids, key=self.book_score, reverse=True)[:n]

    def library_score(self, lib_id, current_day):
        lib = self.libraries[lib_id]
        delta_time = self.D - current_day - lib.signup_time
        n_best_books = self.get_n_best_books(lib, min(delta_time * lib.books_per_day, len(lib.book_ids)))
        sum_of_best_book_scores = sum([self.book_values[book] for book in n_best_books])
        sum_of_best_book_scores /= lib.signup_time
        return sum_of_best_book_scores  # / lib.signup_time ** (1 + lib.signup_time / self.D)

    def get_solution(self):
        remaining_libraries = set(lib.id for lib in self.libraries)
        day = 0
        chosen_libraries = []
        it = 0
        interval = 25  # max(1, int(self.L/1000))
        while len(remaining_libraries) > 0:
            if it % interval == 0:
                library_scores = [(self.library_score(lib_id, day), lib_id) for lib_id in remaining_libraries]
            it += 1
            max_el = max(library_scores, key=lambda x: x[0])
            library_scores.remove(max_el)
            score, lib_id = max_el
            remaining_libraries.remove(lib_id)
            library = self.libraries[lib_id]
            if day + library.signup_time >= self.D:
                break
            day += library.signup_time
            books_to_take = (self.D - day) * library.books_per_day
            sorted_books = sorted(library.book_ids, key=self.book_score, reverse=True)
            for book_id in sorted_books[:books_to_take]:
                for lib_id in self.book_libraries[book_id]:
                    if lib_id != library.id:
                        self.libraries[lib_id].remove_book(book_id)

            for book_id in sorted_books[books_to_take:]:
                self.book_libraries[book_id].remove(library.id)

            library.book_ids = sorted_books[:books_to_take]
            chosen_libraries.append(library)
        return chosen_libraries

    def get_individual_from_solution(self, solution):
        return tuple([lib.id for lib in solution])


class MutationHillClimbingSolver(ProblemSolver):
    def __init__(self, B, L, D, book_values, book_libraries, libraries, individual_scores, neighbourhood_size=10,
                 available_time=100):
        super().__init__(B, L, D, book_values, book_libraries, libraries)
        self.neighbourhood_size = neighbourhood_size
        self.individua_scores = individual_scores
        self.lib_ids = [i for i in range(len(libraries))]
        self.lib_scores = [self.lib_score(lib) for lib in self.libraries]
        self.available_time = available_time

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

    def get_individual(self, start_individual):
        cur_best = copy.copy(start_individual)
        cur_best_score = self.individua_scores[cur_best]
        start_time = time.time()
        progress = []
        cur_state = []
        times = []
        while time.time() - start_time <= self.available_time:
            neighbourhood = []
            n_scores = []
            for i in range(self.neighbourhood_size):
                new_one = self.mutate(cur_best)
                if new_one not in self.individua_scores:
                    new_solution = self.get_solution(new_one)
                    new_score = score_solution(new_solution, self.book_values)
                    self.individua_scores[new_one] = new_score
                else:
                    new_score = self.individua_scores[new_one]
                neighbourhood.append(new_one)
                n_scores.append(new_score)
            new_max_score = max(n_scores)
            new_max_score_index = n_scores.index(new_max_score)
            if new_max_score > cur_best_score:
                cur_best = neighbourhood[new_max_score_index]
                cur_best_score = new_max_score
            progress.append(cur_best_score)
            cur_state.append(new_max_score)
            times.append(time.time() - start_time)
        return cur_best
