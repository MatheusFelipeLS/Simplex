# g++ -O3 -I ./  -lsuitesparseconfig -lumfpack -lamd -lcholmod -lcolamd -lcamd -lccolamd main.cpp -o solve 
g++ -std=gnu++17 -Wall -O3 -o solve main_copy.cpp -I /usr/include/suitesparse -lumfpack -lcholmod -lamd -lsuitesparseconfig
