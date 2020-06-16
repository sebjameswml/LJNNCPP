#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
using namespace std;

//=============VECTOR CREATION=============
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




//=============DISPLAYING VECTORS=============
//print out each item of a vector
void printvect(vector<float>& a){
  for (auto& i : a){
    cout << i << ", ";
  }
  cout << endl;
}
//print out each item of a vector of vectors
void printvectvect(vector<vector<float>>& a){
  for (auto& i : a){
    for (auto& i2 : i){
      cout << i2 << ", ";
    }
    cout << endl;
  }
}
//print out each item of a vector of vectors of vectors
void printvectvectvect(vector<vector<vector<float>>>& a){
  for (auto& i : a){
    for (auto& i2 : i){
      for (auto& i3 : i2){
	cout << i3 << ", ";
      }
      cout << endl;
    }
    cout << "===" << endl;
  }
}




//=============MATH FUNCTIONS=============
//sigmoid function
void sigmoid(float& a,
	     float& tomod){ //tomod will be modified by this function
  tomod = 1/(1+exp(-a));
}
//sigmoid function for vectors
void vectsigmoid(vector<float>& a,
		 vector<float>& vecttomod){ //vecttomod will be modified by this function
  for (int i = 0; i < a.size(); i++){
    sigmoid(a[i], vecttomod[i]);
  }
}
//dot product of two vectors
void dot(vector<float>& a,
	 vector<float>& b,
	 float& product){ //product will be modified by this function
  product = 0;
  for (int i = 0; i < a.size(); i++){
    product += (a[i] * b[i]);
  }
}
void vectadd(vector<float>& a,
	     vector<float>& b,
	     vector<float>& product){ //product will be modified by this function
  for (int i = 0; i < a.size(); i ++){
    product[i] = a[i] + b[i];
  }
}



//=============NN FUNCS=============
void feedforwards(
		  vector<vector<vector<float>>>& weights,
		  vector<vector<float>>& biases,
		  vector<vector<float>>& activations,
		  vector<vector<float>>& presigactivations
		  ){

}



//=============MAIN=============
int main(){
  //set shape of network
  int sizes[3] = {2,3,2};
  //create weights
  vector<vector<vector<float>>> weights;
  weights.resize(sizeof(sizes)/sizeof(*sizes)-1);
  //fill weights with random floats
  for (int i = 0; i < sizeof(sizes)/sizeof(*sizes)-1; i++){
    weights[i] = randvectvect(sizes[i+1], sizes[i]);
  }
  //create biases
  vector<vector<float>> biases;
  biases.resize(sizeof(sizes)/sizeof(*sizes)-1);
  //fill biases with random floats
  for (int i = 0; i < sizeof(sizes)/sizeof(*sizes)-1; i++){
    biases[i] = randvect(sizes[i+1]);
  }
  //print out weights and biases to verify the above code works
  cout << "weights:" << endl;
  printvectvectvect(weights);
  cout << "biases:" << endl;
  printvectvect(biases);
  vector<float> x = {1,2,3};
  vector<float> y = {1,2,3};
  float z = 0;
  cout << "-----" << endl;
  dot(x,y,z);
  vectsigmoid(x, y);
  printvect(y);
  cout << "z " << z << endl;
  return 0;
}
