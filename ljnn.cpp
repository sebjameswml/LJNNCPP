#include <iostream>
#include <vector>
#include <numeric>
#include <random>
using namespace std;

vector<float> randvect(int x){
  random_device rd;
  mt19937 randf(rd());
  uniform_real_distribution<> dist(0,1);

  //create a vector of specified size
  vector<float> v;
  v.resize(x);
  for (auto& i : v){
    i = dist(randf);
  }
  return v;
}

vector<vector<float>> randvectvect(int x, int y){

  random_device rd;
  mt19937 randf(rd());
  uniform_real_distribution<> dist(0,1);
  
  //create vector of vectors of specified size
  vector<vector<float>> v;
  v.resize(x);
  for (auto& i : v){
    i.resize(y);
  }
  //make each item a random number
  for (auto& i : v){
    for (auto& i2 : i){
      i2 = dist(randf);
    }
  }
  //return the vector of vectors
  return v;
}









int main(){
  //size of the network
  int sizes[4] = {2,3,3,2};

  
  //create weights
  vector<vector<vector<float>>> weights;
  weights.resize(sizeof(sizes)/sizeof(*sizes)-1);
  
  //fill weights with random floats
  int i = 0;
  while (i < sizeof(sizes)/sizeof(*sizes)-1) {
    weights[i] = randvectvect(sizes[i], sizes[i+1]);
    i+=1;
}


  
  //create biases
  vector<vector<float>> biases;
  biases.resize(sizeof(sizes)/sizeof(*sizes)-1);
  i = 0;
  while (i < sizeof(sizes)/sizeof(*sizes)-1) {
    biases[i] = randvect(sizes[i+1]);
    i+=1;
  }







  
  //print weights
  cout << "weights:" << endl;
  for (auto& i : weights){
    for (auto& i2 : i){
      for (auto& i3 : i2) {
	cout << i3 << ", ";
      }
      cout << "\n";
    }
    cout << "" << endl;
  }



  
  //print biases
  cout << "biases:" << endl;
  for (auto& i : biases){
    for (auto& i2 : i){
      cout << i2 << ", ";
    }
    cout << "\n";
  }
}
