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

//derivative of sigmoid
float sigmoidderiv(float x){
  return sigmoid(x)*(1-sigmoid(x));
}

//derivative of sigmoid but for a whole vector
vector<float> vectsigmoidderic(vector<float> x){
  vector<float> product;
  product.resize(x.size());
  for (int i = 0; i < x.size(); i++){
    product[i] = sigmoidderiv(x[i]);
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
void feedforwards(const vector<vector<vector<float>>>& weights,
		  const vector<vector<float>>& biases,
		  const vector<float>& input,
		  vector<vector<float>>& activations,
		  vector<vector<float>>& presigactivations){
  vector<float> a = input;
  vector<float> dp;
  activations[0] = a;
  presigactivations[0] = a;
  for (int i = 0; i < 2; i++){
    dp.resize(biases[i].size());
    for (auto& idp : dp){
      idp = dot(weights[i][idp], a);
    }
    a = addvect(dp, biases[i]);
    presigactivations[i+1] = a;
    a = vectsigmoid(a);
    activations[i+1] = a;
  }
}

//get error of last layer
vector<float> getlasterror(const vector<vector<float>>& activations,
			   const vector<vector<float>>& presigactivations,
			   const vector<float>& desiredoutput
			   ){
  vector<float> deltalast;
  deltalast.resize(activations.back().size());
  for (int i = 0; i < activations.back().size(); i++){
    deltalast[i] = (activations.back()[i] - desiredoutput[i])*sigmoidderiv(presigactivations.back()[i]);
  }
  return deltalast;
}
//get all errors
vector<vector<float>> geterrors(const vector<vector<float>>& activations,
				const vector<vector<float>>& presigactivations,
				const vector<float>& desiredoutput
				){
  vector<vector<float>>delta;
  delta.resize(activations.size());
  delta.back() = getlasterror(activations, presigactivations, desiredoutput);
  
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

  vector<vector<float>> activations;
  vector<vector<float>> presigactivations;
  activations.resize(biases.size()+1);
  presigactivations.resize(biases.size()+1);
  
  feedforwards(weights, biases, inp, activations, presigactivations);
  cout << "feedforwards:" << endl;
  for (auto& i1 : activations){
    for (auto& i2 : i1){
      cout << i2 << ", ";
    }
    cout << endl;
  }

  getlasterror(activations, presigactivations, {1,1});
  return 0;
}
