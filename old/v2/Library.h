//
// Created by Janek_PC on 23.12.2020.
//

#include <vector>
#ifndef UOPO2_LIBRARY_H
#define UOPO2_LIBRARY_H


struct Library {
    int numBooks{};
    int daysToSignUp{};
    int booksPerDay{};
    int value{};
    int libraryNumber{};
    std::vector<int> books;
    std::vector<int> booksToScan;
};

Library findLibrary(int currentDay, std::vector<Library> &libraries, int dead_line, const std::vector<int> &book_scores);
void removeDuplicates(std::vector<Library> &libraries, std::vector<int> &booksToRemove);
#endif //UOPO2_LIBRARY_H
