#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
using namespace std;

//hadamard product of two vectors
vector<float> hadamard(const vector<float>& a, const vector<float>& b){
  vector<float> product;
  product.resize(a.size());
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] * b[i];
  }
  return product;
}

//hadamard product of two vectors of vectors
vector<vector<float>> vvhadamard(const vector<vector<float>>& a, const vector<vector<float>>& b){
  vector<vector<float>> product;
  product.resize(a.size());
  for (int i = 0; i < a.size(); i++){
    product[i].resize(a[i].size());
  }
  for (int i = 0; i < a.size(); i++){
    product[i] = hadamard(a[i], b[i]);
  }
  return product;
}

//transpose vector of vectors
vector<vector<float>> transpose(const vector<vector<float>>& a){
  vector<vector<float>> result;
  result.resize(a[0].size());
  for (int i = 0; i < result.size(); i++){
    result[i].resize(a.size());
  }
  for (vector<int>::size_type i = 0; i < a[0].size(); i++){
    for (vector<int>::size_type j = 0; j < a.size(); j++){
      result[i][j] = a[j][i];
    }
  }
  return result;
}

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
vector<float> vectsigmoidderiv(vector<float> x){
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

//subtract vector b from vector a
vector<float> subvect(vector<float> a, vector<float> b){
  vector<float> product;
  product.resize(a.size());
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] - b[i];
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
				const vector<float>& desiredoutput,
				const vector<vector<vector<float>>>& weights
				){
  //create all the vectors we need
  vector<vector<float>>delta;
  vector<float> x;
  vector<float> y;
  vector<vector<float>> tw;
  vector<float> product;
  //give delta correct size
  delta.resize(activations.size());
  for (int i = 0; i < activations.size(); i++){
    delta[i].resize(activations[i].size());
  }
  //get delta for output neurons
  delta.back() = getlasterror(activations, presigactivations, desiredoutput);
  //backpropogate
  for (int i = delta.size()-2; i > -1; i--){
    //transpose weights[i]
    tw = transpose(weights[i]);
    //give product the correct size
    product.resize(activations[i].size());
    //matrix multiplication of transposed weights[i] and delta[i+1]
    for (int mm = 0; mm < activations[i].size(); mm++){
      product[mm] = dot(tw[mm], delta[i+1]);
    }
    //getting y is easy
    y.resize(activations[i].size());
    y = vectsigmoidderiv(presigactivations[i]);
    //do the final hadamard product
    delta[i] = hadamard(product, y);
  }
  return delta;
}
//Mean Squared Error
float MSE(vector<float> outputactivations, vector<float> desiredoutput){
  float cost = 0;
  for (int i = 0; i < desiredoutput.size(); i++){
    float x = outputactivations[i] - desiredoutput[i];
    cost += pow(x, 2.0f);
  }
  return cost;
}
//matrix multiplication but only 1D
vector<float> vecmul(vector<float>& a, vector<float>& b){
  vector<float> product;
  float dotproduct;
  product.resize(a.size());
  for (int i = 0; i < a.size(); i++){
    dotproduct = 0;
    for (int dp = 0; dp < b.size(); dp++){
      dotproduct += a[i] * b[dp];
    }
  }
  return product;
}
//another type of matrix multiplication
vector<float> singlevecmul(vector<float> a, float b){
  for(auto& i : a){
    i *= b;
  }
  return a;
}
//main
int main(){
  //size of the network
  int sizes[3] = {10,10,10};
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
  //create some required vectors
  vector<float> inp = {1,1,1,1,1,1,1,1,1,1};
  vector<vector<float>> activations;
  vector<vector<float>> presigactivations;
  vector<vector<float>> delta;
  //resize some of the vectors
  activations.resize(sizeof(sizes)/sizeof(*sizes));
  presigactivations.resize(sizeof(sizes)/sizeof(*sizes));
  //train
  float eta = 5; //learning rate
  vector<float> desiredout = {1,1,1,1,1,1,1,1,1,1};
  for (int i = 0; i < 50; i++){
    feedforwards(weights, biases, inp, activations, presigactivations);
    delta = geterrors(activations, presigactivations, desiredout, weights);
    cout << "cost" << MSE(activations.back(), desiredout) << endl;
    //update biases
    for (int bi = 0; bi < biases.size(); bi++){
      biases[bi] = singlevecmul(subvect(biases[bi], delta[bi+1]), eta);
    }
    //update weights
    for (int wi = 0; wi < weights.size(); wi++){
      //weights[wi] = (weights[wi], vecmul(activations[wi+1], delta[wi]));
      for (int i = 0; i < weights[wi].size(); i++){
        weights[wi][i] = singlevecmul(vecmul(activations[wi], delta[wi]), eta);
      }
    }
  }
  return 0;
}
