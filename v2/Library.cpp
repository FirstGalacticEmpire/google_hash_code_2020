//
// Created by Janek_PC on 23.12.2020.
//
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "Library.h"
#include <iostream>
#include <vector>
#include "Library.h"
#include <algorithm>
#include <algorithm>
#include <unordered_set>
#include "Library.h"

using namespace std;
bool compareLibrariesByValue(const Library &a, const Library &b) {
    return a.value > b.value;
}
Library findLibrary(int currentDay, vector<Library> &libraries, int dead_line, const vector<int> &book_scores) {
    for (auto &library : libraries) {
        int activeDays = dead_line - currentDay - library.daysToSignUp;
        int booksCount = min(activeDays * library.booksPerDay, library.numBooks);

        int value = 0;
        int counter = 0;
        library.booksToScan.clear();

        for (auto &book : library.books) {
            //cout<<book<<" ";
            library.booksToScan.push_back(book);
            value += book_scores.at(book);
            counter += 1;
            if (counter == booksCount -1) {//possibly booksCount-1
                break;
            }

        }
        library.value = value;

        //cout<<value<<endl;
    }

    sort(libraries.begin(), libraries.end(), compareLibrariesByValue);

    return libraries.at(0);
}

void removeDuplicates(vector<Library> &libraries, vector <int> &booksToRemove) {
    for (auto &library : libraries) {

        std::unordered_set<int> set;
        for (auto &book: booksToRemove) {
            set.insert(book);
        }
        library.books.erase(remove_if(library.books.begin(), library.books.end(), [&set](int book) {
            if (set.count(book) == 0) {
                return false;
            } else {
                return true;
            }
        }), library.books.end());

    }
}