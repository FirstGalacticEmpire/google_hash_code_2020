//
// Created by Janek_PC on 15.12.2020.
//
#include <vector>

#ifndef OPOP_LIBRARY_H
#define OPOP_LIBRARY_H


struct Library {
    int numBooks{};
    int daysToSignUp{};
    int scanPerDay{};
    int score{};
    std::vector<int> books;
};

int findLibrary(int currentDay, std::vector<Library> &libraries, int B, int L, int dead_line);
#endif //OPOP_LIBRARY_H
