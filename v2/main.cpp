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

using namespace std;

vector<int> book_scores;

vector<int> makeVector(string &line_to_parse, string &delimiter) {
    vector<int> aVector;
    size_t pos = 0;
    string token;
    do {
        pos = line_to_parse.find(delimiter);
        token = line_to_parse.substr(0, pos);
        aVector.push_back(stoi(token, nullptr, 0)); //0 means it automatically specifies type
        line_to_parse.erase(0, pos + delimiter.length());
    } while (pos != string::npos);

    return aVector;
}

bool compareBooksByScore(const int &a, const int &b) {
    return book_scores.at(a) > book_scores.at(b);
}

int main() {
    //ifstream file("f_libraries_of_the_world.txt");
    ifstream file("d_tough_choices.txt");
    int B = 0, L = 0, dead_line = 0;
    vector<Library> libraries;

    if (file.is_open()) {
        string line;
        string delimiter = " ";

        getline(file, line);


        // Parsing first line
        vector<int> b_l_d = makeVector(line, delimiter);

        dead_line = b_l_d.at(2);
        L = b_l_d.at(1);
        B = b_l_d.at(0);

        b_l_d.clear();

        getline(file, line);
        book_scores = makeVector(line, delimiter);


        for (int a = 0; a < L; a++) {
            getline(file, line);

            vector<int> a_vector = makeVector(line, delimiter);

            struct Library temp;
            temp.booksPerDay = a_vector.at(2);
            temp.daysToSignUp = a_vector.at(1);

            temp.numBooks = a_vector.at(0);
            temp.libraryNumber = a;
            a_vector.clear();

            getline(file, line);

            temp.books = makeVector(line, delimiter);
            libraries.push_back(temp);
            a_vector.clear();

        }


        file.close();
    } else {
        throw "assadasd";
    }

    for (auto &library : libraries) {
        sort(library.books.begin(), library.books.end(), compareBooksByScore);
    }
    int result=0;
    Library library = findLibrary(0, libraries, dead_line, book_scores);
    for (int i = 0; i < dead_line; i++) {
        library.daysToSignUp -= 1;
        if (library.daysToSignUp == 0) {

            cout << libraries.at(0).libraryNumber << " " << libraries.at(0).booksToScan.size() << endl;
            for (auto &book : library.booksToScan) {
                cout << book << " ";
//                cout<<"asds"<<book_scores.at(book)<<"asd";
                result = result + book_scores.at(book);
            }
            cout<<endl;
            libraries.erase(libraries.begin());
            if (!libraries.empty()) {
                removeDuplicates(libraries, library.booksToScan);
                library = findLibrary(i, libraries, dead_line, book_scores);
            } else {
                break;
            }
        }


    }
    cout<<endl<<result;


}
