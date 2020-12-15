#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;


vector<int> makeVector(string &line_to_parse, string &delimiter) {
    vector<int> aVector;
    size_t pos = 0;
    string token;
    do {
        pos = line_to_parse.find(delimiter);
        token = line_to_parse.substr(0, pos);
        aVector.push_back(stoi(token, nullptr, 0));
        line_to_parse.erase(0, pos + delimiter.length());
    } while (pos != string::npos);

    return aVector;
}

struct Library {
    int numBooks{};
    int singUp{};
    int shipNum{};
    vector<int> books;
};

int main() {
    ifstream file("a_example.txt");
    int B, L, dead_line;


    if (file.is_open()) {
        string line;
        string delimiter = " ";

        getline(file, line);

        cout << line << endl;

        // Parsing first line
        vector<int> b_l_d = makeVector(line, delimiter);
        dead_line = b_l_d.at(2);
        L = b_l_d.at(1);
        B = b_l_d.at(0);
        //clearing memory
        b_l_d.clear();

        getline(file, line);
        vector<int> stats = makeVector(line, delimiter);

        vector<Library> libraries;

        for (int a = 0; a < L; a++) {
            getline(file, line);

            vector<int> a_vector = makeVector(line, delimiter);

            struct Library temp;
            temp.shipNum = a_vector.at(2);
            temp.singUp = a_vector.at(1);
            temp.numBooks = a_vector.at(0);
            a_vector.clear();

            getline(file, line);

            a_vector = makeVector(line, delimiter);
            temp.books = a_vector;
            libraries.push_back(temp);
            a_vector.clear();

        }
        cout<< libraries.at(1).numBooks;
        file.close();
    }




}
