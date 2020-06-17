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
//sigmoid prime
void sigmoidprime(float& a,
		  float& tomod){ //tomod will be modified by this function
  sigmoid(a, tomod); //set tomod to sigmoid(a)
  tomod = tomod*(1-tomod); //this may not work
}
//sigmoid function for vectors
void vectsigmoid(vector<float>& a,
		 vector<float>& vecttomod){ //vecttomod will be modified by this function
  for (int i = 0; i < a.size(); i++){
    sigmoid(a[i], vecttomod[i]);
  }
}
//sigmoid prime for vectors
void vectsigmoidprime(vector<float>& a,
		      vector<float>& vecttomod){ //vecttomod will be modified by this function
  for (int i = 0; i < a.size(); i++){
    sigmoidprime(a[i], vecttomod[i]);
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
//adds two vectors of the same size
void vectadd(vector<float>& a,
	     vector<float>& b,
	     vector<float>& product){ //product will be modified by this function
  for (int i = 0; i < a.size(); i ++){
    product[i] = a[i] + b[i];
  }
}
//sets product to vector a - vector b
void vectsub(vector<float>& a,
	     vector<float>& b,
	     vector<float>& product){ //product will be modified by this function
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] - b[i];
  }
}
//hadamard product for single vectors
void hadamard(vector<float>& a,
	      vector<float>& b,
	      vector<float>& product //all 3 of these vectors must have identical sizes.
	      ){
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] * b[i];
  }
}




//=============NN FUNCS=============
//runs a forwards pass through the network
void feedforwards(
		  vector<vector<vector<float>>>& weights,
		  vector<vector<float>>& biases,
		  vector<vector<float>>& activations,
		  vector<vector<float>>& presigactivations,
		  vector<float>& x
		  ){
  for (int layer = 0; layer < activations.size()-1; layer++){
    x.resize(activations[layer+1].size());
    for (int neuron = 0; neuron < activations[layer].size(); neuron++){
      dot(weights[layer][neuron], activations[layer], x[neuron]); //sets x to the dot product of weights[layer] and activations[layer]
    }
    vectadd(x, biases[layer], presigactivations[layer+1]); //sets presigactivations[layer] to x+biases[layer]
    vectsigmoid(presigactivations[layer+1], activations[layer+1]); //sets activations[layer+1] to the (vect)sigmoid of presigactivations[layer]
  }
}
//backpropogation //geterrors(weights, biases, activations, presigactivations, delta, desiredoutput, x, y, z);
void geterrors(
	       vector<vector<vector<float>>>& weights,
	       vector<vector<float>>& biases,
	       vector<vector<float>>& activations,
	       vector<vector<float>>& presigactivations,
	       vector<vector<float>>& delta,
	       vector<float>& desiredoutput,
	       vector<float>& x,
	       vector<float>& y,
	       vector<float>& z
	       ){
  
  vectsub(activations.back(), desiredoutput, x); //make x output-desiredoutput
  vectsigmoidprime(presigactivations.back(), y); //make y = sigmoid prime of output presigactivations
  hadamard(x, y, z); //make z = the hadamard product of x and y
  delta.back() = z;
}




//=============MAIN=============
int main(){
  //set shape of network
  int sizes[3] = {2,3,4};
  //create weights
  vector<vector<vector<float>>> weights;
  weights.resize(sizeof(sizes)/sizeof(*sizes)-1);
  //fill weights with random floats
  for (int i = 0; i < sizeof(sizes)/sizeof(*sizes)-1; i++){
    weights[i] = randvectvect(sizes[i], sizes[i+1]);
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
  cout << "=======" << endl;
  
  //create activations vector
  vector<vector<float>> activations;
  //allocate memory for activations
  activations.resize(sizeof(sizes)/sizeof(*sizes));
  for (int i = 0; i < sizeof(sizes)/sizeof(*sizes); i++){
    activations[i].resize(sizes[i]);
  }
  //create pre-sigmoid activations vector
  vector<vector<float>> presigactivations;
  //allocate memory for pre-sigmoid activations vector
  presigactivations.resize(sizeof(sizes)/sizeof(*sizes));
  for (int i = 0; i < sizeof(sizes)/sizeof(*sizes); i++){
    presigactivations[i].resize(sizes[i]);
  }
  //create vectors used to hold data, defined out-of function for possible optimisation
  vector<float> ffx;
  //set input activations:
  activations[0] = {1,1};
  //run a single feedforwards pass
  feedforwards(weights, biases, activations, presigactivations, ffx);
  //print activations
  cout << "activations:" << endl;
  printvectvect(activations);


  //vectors required for backprop
  vector<float> bpx;
  vector<float> bpy;
  vector<float> bpz;
  vector<vector<float>> delta = {{0},{0},{0}};
  vector<float> desiredoutput = {1,1,1,1};
  //these resizes are required for the geterrors() function
  bpx.resize(desiredoutput.size());
  bpy.resize(desiredoutput.size());
  bpz.resize(desiredoutput.size());
  
  geterrors(weights, biases, activations, presigactivations, delta, desiredoutput, bpx, bpy, bpz);

  cout << "delta:" << endl;
  printvectvect(delta);
  return 0;
}
