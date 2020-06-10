#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
using namespace std;

//good old sigmoid function
float sigmoid(float z){
  return 1/(1+exp(-z));
}

//apply the sigmoid function to each item in a vector
vector<float> vectsigmoid(vector<float> z){
  vector<float> product;
  product.resize(z.size());
  for (int i = 0; i < z.size(); i++){
    product[i] = sigmoid(z[i]);
  }
  return product;
}

//dot product of two vectors
float dot(vector<float> a, vector<float> b){  
  float product = 0;
  for (int i = 0; i < a.size(); i++){
    product += (a[i] * b[i]);
  }
  return product;
}

//add two vectors
vector<float> addvect(vector<float> a, vector<float> b){
  vector<float> product;
  product.resize(a.size());
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] + b[i];
  }
  return product;
}

//create a vector of a specified size and fill it with random floats
vector<float> randvect(int& x){
  random_device rd;
  mt19937 randf(rd());
  uniform_real_distribution<> dist(0,1);
  //create a vector of specified size
  vector<float> v;
  v.resize(x);
  //fill vector with random floats
  for (auto& i : v){
    i = dist(randf);
  }
  return v;
}

//create a vector of vectors of a specified dimension and fill them with random floats
vector<vector<float>> randvectvect(int& x, int& y){
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

//neural network feedforwards
vector<vector<float>> feedforwards(
		   const vector<vector<vector<float>>>& weights,
		   const vector<vector<float>>& biases,
		   const vector<float>& input
		   ){
  vector<float> a = input;
  vector<float> dp;
  vector<vector<float>> out;
  out.resize(biases.size()+1);
  out[0] = a;
  for (int i = 0; i < 2; i++){
    dp.resize(biases[i].size());
    for (auto& idp : dp){
      idp = dot(weights[i][idp], a);
      }
    a = addvect(dp, biases[i]);
    a = vectsigmoid(a);
    out[i+1] = a;
  }
  return out;
} 

//main
int main(){
  //size of the network
  int sizes[3] = {2,3,2};

  //create weights
  vector<vector<vector<float>>> weights;
  weights.resize(sizeof(sizes)/sizeof(*sizes)-1);
  
  //fill weights with random floats
  int i = 0;
  while (i < sizeof(sizes)/sizeof(*sizes)-1) {
    weights[i] = randvectvect(sizes[i+1], sizes[i]);
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

  vector<float> inp = {1,1};
  cout << "=================" << endl;
  vector<vector<float>> activations = feedforwards(weights, biases, inp);
  cout << "feedforwards:" << endl;
  for (auto& i1 : activations){
    for (auto& i2 : i1){
      cout << i2 << ", ";
    }
    cout << endl;
  }
  return 0;
}
