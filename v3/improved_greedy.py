import numpy as np, matplotlib, time, copy, random, math

from ortools.graph import pywrapgraph

directory = '../data/'
file_paths = ['a_example.txt', 'b_read_on.txt', 'c_incunabula.txt', 'd_tough_choices.txt', 'e_so_many_books.txt',
              'f_libraries_of_the_world.txt']


class Library():
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


def process_file(filePath):
    with open(filePath, 'r') as file:
        content = file.read().split('\n')[:-1];
        B, L, D = content[0].split()
        B, L, D = int(B), int(L), int(D)
    book_libraries = [set() for i in range(B)]
    bookValues = [int(n) for n in content[1].split()]
    libraries = []
    for i in range(L):
        N, T, M = content[2 + 2 * i].split()
        N, T, M = int(N), int(T), int(M)
        book_ids = [int(id) for id in content[2 + 2 * i + 1].split()]
        library = Library(i, N, T, M)
        for book_id in book_ids:
            book_libraries[book_id].add(i)
            library.add_book(book_id)
        libraries.append(library)
    return ((B, L, D), bookValues, book_libraries, libraries)


def check_solution(D, libraries):
    days = 0
    prev_books = set()
    for library in libraries:
        days += library.signup_time
        if len(library.book_ids) > (D - days) * library.books_per_day:
            print("what", len(library.book_ids), (D - days) * library.books_per_day)
        assert (len(library.book_ids) == len(set(library.book_ids)))
        assert (not any([(book in prev_books) for book in library.book_ids]))
        prev_books.update(library.book_ids)
    assert (days < D)


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
        self.book_libraries = copy.copy(book_libraries)

    def book_score(self, book):
        return self.book_values[book]

    def get_n_best_books(self, lib, n):
        return sorted(lib.book_ids, key=self.book_score, reverse=True)[:n]

    def get_solution(self, selected_lib_ids=None):
        if not selected_lib_ids:
            selected_lib_ids = self.get_individual()
        local_book_values = copy.copy(self.book_values)
        day = 0
        selected_libraries = [copy.copy(self.libraries[i]) for i in selected_lib_ids]
        it = 0;
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


class GreedyIntervalSolver:
    def __init__(self, B, L, D, book_values, book_libraries, libraries):
        self.B, self.L, self.D = B, L, D
        self.book_values = book_values
        self.libraries = libraries
        self.book_libraries = copy.copy(book_libraries)

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
        libraries = [copy.copy(library) for library in self.libraries]
        remaining_libraries = set(lib.id for lib in libraries)
        day = 0
        chosen_libraries = []
        it = 0
        while len(remaining_libraries) > 0:
            if it % 100 == 0:
                library_scores = [(self.library_score(lib_id, day), lib_id) for lib_id in remaining_libraries]
            it += 1
            max_el = max(library_scores, key=lambda x: x[0])
            library_scores.remove(max_el)
            score, lib_id = max_el
            remaining_libraries.remove(lib_id)
            library = libraries[lib_id]
            if day + library.signup_time >= self.D:
                break
            day += library.signup_time
            books_to_take = (self.D - day) * library.books_per_day
            sorted_books = sorted(library.book_ids, key=self.book_score, reverse=True)
            for book_id in sorted_books[:books_to_take]:
                for lib_id in self.book_libraries[book_id]:
                    if lib_id != library.id:
                        libraries[lib_id].remove_book(book_id)

            for book_id in sorted_books[books_to_take:]:
                self.book_libraries[book_id].remove(library.id)

            library.book_ids = sorted_books[:books_to_take]
            chosen_libraries.append(library)
        return chosen_libraries

    def get_individual_from_solution(self, solution):
        return tuple([lib.id for lib in solution])


if __name__ == "__main__":
    sum_score = 0
    for file_path in file_paths:
        (B, L, D), book_values, book_counts, libraries = process_file(file_path)
        solver = GreedyIntervalSolver(B, L, D, book_values, book_counts, libraries)
        solution = solver.get_solution()
        # print(solution)
        check_solution(D, solution)
        score = score_solution(solution, book_values)
        print("S: ", score)
        sum_score += score

    print(sum_score)
