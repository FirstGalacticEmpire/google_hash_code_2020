//
// Created by Janek_PC on 15.12.2020.
//
#include <iostream>
#include <vector>
#include "Library.h"
#include <algorithm>

bool compareByScore(const Library &a, const Library &b)
{
    return a.score < b.score;
}

using namespace std;
int findLibrary(int currentDay, vector<Library> &libraries, int B, int L, int dead_line){
    int score;
    for(auto & library : libraries) {
        int activeDays = dead_line - currentDay - library.scanPerDay;
        score = min(activeDays, library.numBooks);
        score = score/library.daysToSignUp;
        library.score = score;
    }

    sort(libraries.begin(), libraries.end(), compareByScore); // xd

    for(auto & library : libraries) {
        cout<<library.daysToSignUp;
    }

    return 0;
}