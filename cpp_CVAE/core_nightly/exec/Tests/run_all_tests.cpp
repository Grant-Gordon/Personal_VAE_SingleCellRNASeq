//run_all_tests.cpp
#include <iostream>

void test_DenseLinear();
void test_SparseLinear();
void test_RELULayer();


int main() {
    std::cout << "Running unit Tests...\n";

    test_DenseLinear();
    test_SparseLinear();
    test_RELULayer();

    std::cout<< "All tests completed\n'";
    return 0;
}
