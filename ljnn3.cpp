#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>
using namespace std;




//=============DISPLAYING VECTORS=============
//print out each item of a vector
void printvect(const vector<float>& a){
  for (auto& i : a){
    cout << i << ", ";
  }
  cout << endl;
}
//print out each item of a vector of vectors
void printvectvect(const vector<vector<float>>& a){
  for (auto& i : a){
    for (auto& i2 : i){
      cout << i2 << ", ";
    }
    cout << endl;
  }
}
//print out each item of a vector of vectors of vectors
void printvectvectvect(const vector<vector<vector<float>>>& a){
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
float sigmoid(const float& a,
	     float& tomod){ //tomod will be modified by this function
  tomod = 1/(1+exp(-a));
  return tomod;
}
//sigmoid prime
float sigmoidprime(const float& a,
		  float& tomod){ //tomod will be modified by this function
  sigmoid(a, tomod);
  tomod = tomod*(1-tomod);
  return tomod;
}
//sigmoid function for vectors
vector<float> vectsigmoid(const vector<float>& a,
		 vector<float>& vecttomod){ //vecttomod will be modified by this function
  for (int i = 0; i < a.size(); i++){
    sigmoid(a[i], vecttomod[i]);
  }
  return vecttomod;
}
//sigmoid prime for vectors
vector<float> vectsigmoidprime(const vector<float>& a,
		      vector<float>& vecttomod){ //vecttomod will be modified by this function
  for (int i = 0; i < a.size(); i++){
    sigmoidprime(a[i], vecttomod[i]);
  }
  return vecttomod;
}
//dot product of two vectors
float dot(const vector<float>& a,
	 const vector<float>& b,
	 float& product){ //product will be modified by this function
  if (a.size() != b.size()){
    throw runtime_error("dot product received vectors of different sizes");
  }
  product = 0;
  for (int i = 0; i < a.size(); i++){
    product += (a[i] * b[i]);
  }
  return product;
}
//adds two vectors of the same size
vector<float> vectadd(const vector<float>& a,
	     const vector<float>& b,
	     vector<float>& product){ //product will be modified by this function
  for (int i = 0; i < a.size(); i ++){
    product[i] = a[i] + b[i];
  }
  return product;
}
//sets product to vector a - vector b
vector<float> vectsub(const vector<float>& a,
	     const vector<float>& b,
	     vector<float>& product){ //product will be modified by this function
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] - b[i];
  }
  return product;
}
//hadamard product for single vectors
vector<float> hadamard(const vector<float>& a,
	      const vector<float>& b,
	      vector<float>& product //all 3 of these vectors must have identical sizes.
	      ){
  for (int i = 0; i < a.size(); i++){
    product[i] = a[i] * b[i];
  }
  return product;
}
//transpose vector of vectors
vector<vector<float>> transpose(const vector<vector<float>>& a, vector<vector<float>>& transposed){ //transposed will be modified by this function
  transposed.resize(a[0].size());
  for (int i = 0; i < transposed.size(); i++){
    transposed[i].resize(a.size());
  }

  for (vector<int>::size_type i = 0; i < a[0].size(); i++){
    for (vector<int>::size_type j = 0; j < a.size(); j++){
      transposed[i][j] = a[j][i];
    }
  }
  return transposed;
}




//=============MNIST LOADER=============
int chars_to_int (const char* buf){
  int rtn = (buf[3]&0xff) | (buf[2]&0xff)<<8 | (buf[1]&0xff)<<16 | (buf[0]&0xff)<<24;
  return rtn;
}

vector<int> loadlabels(){
  ifstream t_lab;
  t_lab.open("train-labels-idx1-ubyte", ios::in | ios::binary);
  //process file data
  char buff[4];
  t_lab.read(buff, 4);
  int magic_labs = chars_to_int (buff);
  t_lab.read(buff, 4);
  int n_labs = chars_to_int (buff);
  cout << "n_labs " << n_labs << endl;
  //create and resize labels vector
  vector<int> labels;
  labels.resize(n_labs);
  //read values into labels vector
  char lbuff[1];
  for (int label = 0; label < n_labs; label++){
    t_lab.read(lbuff, 1);
    unsigned char uc = lbuff[0];
    labels[label] = uc;
  }
  return labels;
}

vector<vector<float>> loadimages(){
  ifstream t_img;
  t_img.open("train-images-idx3-ubyte", ios::in | ios::binary);
  //process file data
  char buff[4];
  t_img.read(buff, 4);
  int magic_imgs = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_imags = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_rows = chars_to_int (buff);
  t_img.read(buff, 4);
  int n_cols = chars_to_int (buff);
  cout << "n_imags " << n_imags << endl;
  //create and resize images vector
  vector<vector<float>> images;
  images.resize(n_imags);
  for (int i = 0; i < n_imags; i++){
    images[i].resize(n_cols*n_rows);
  }
  //read values into images vector
  char pbuff[1];
  for (int image = 0; image < n_imags; image++){
    for (int pixel = 0; pixel < (n_rows*n_cols); pixel++){
      t_img.read(pbuff, 1);
      unsigned char uc = pbuff[0];
      images[image][pixel] = uc;
    }
  }
  return images;
}




//=============NN FUNCS=============
//runs a forwards pass through the network //feedforwards(weights, biases, activations, presigactivations, ffx);
void feedforwards(
		  const vector<vector<vector<float>>>& weights,
		  const vector<vector<float>>& biases,
		  vector<vector<float>>& _activations,
		  vector<vector<float>>& _presigactivations,
		  vector<float>& fx
		  ){
  _presigactivations[0] = _activations[0];
  for (int layer = 0; layer < _activations.size()-1; layer++){

    for (auto& i : fx){
      i = 0;
    }

    for (int neuron = 0; neuron < _activations[layer+1].size(); neuron++){
      for (int dot = 0; dot < _activations[layer].size(); dot++){
	fx[neuron] += _activations[layer][dot] * weights[layer][neuron][dot];
      }
    }
    printvect(fx);
    cout << "fxfxfxfx" << endl;
    vectadd(fx, biases[layer], _presigactivations[layer+1]); //sets presigactivations[layer+1] to fx+biases[layer]
    vectsigmoid(_presigactivations[layer+1], _activations[layer+1]); //sets activations[layer+1] to the (vect)sigmoid of presigactivations[layer]
  }

}
//backpropogation //geterrors(weights, biases, activations, presigactivations, delta, desiredoutput, x, y, z, sigprime, matmulproduct, transposedweights);
void geterrors(
	       const vector<vector<vector<float>>>& weights,
	       const vector<vector<float>>& biases,
	       const vector<vector<float>>& activations,
	       const vector<vector<float>>& presigactivations,
	       vector<vector<float>>& _delta,
	       const vector<float>& desiredoutput,
	       vector<float>& x,
	       vector<float>& y,
	       vector<float>& z,
	       vector<float>& sigprime,
	       vector<float>& matmulproduct,
	       vector<vector<float>>& _transposedweights
	       ){

  vectsub(activations.back(), desiredoutput, x); //make x output-desiredoutput
  vectsigmoidprime(presigactivations.back(), y); //make y = sigmoid prime of output presigactivations
  hadamard(x, y, z); //make z = the hadamard product of x and y
  _delta.back() = z;
  //go backwards through the network and calculate delta for the remaining layers
  for (int layer = _delta.size()-2; layer > -1; layer--){
    matmulproduct.resize(_delta[layer].size());
    sigprime.resize(_delta[layer].size());
    transpose(weights[layer], _transposedweights);
    for (int mm = 0; mm < activations[layer+1].size(); mm++){
      dot(_delta[layer+1], _transposedweights[mm], matmulproduct[mm]);
    }
    vectsigmoidprime(presigactivations[layer+1], sigprime);
    hadamard(matmulproduct, sigprime, _delta[layer]);
  }
}
//mean squared error cost function
float MSE(vector<float> outputactivations, vector<float> desiredoutput){
  float cost = 0;
  for (int i = 0; i < desiredoutput.size(); i++){
    float x = outputactivations[i] - desiredoutput[i];
    cost += pow(x, 2.0f);
  }
  return cost;
}




//=============MAIN=============
int main(){
  //set up RNG
  random_device rd;
  mt19937 randf(rd());
  uniform_real_distribution<> dist(0,1);

  //set shape of network
  int sizes[3] = {784,32,10};
  int largest = *max_element(sizes, sizes+sizeof(sizes)/sizeof(*sizes)); //this needs changing so it doesnt count the input layer
  //create weights
  vector<vector<vector<float>>> weights;
  weights.resize(sizeof(sizes)/sizeof(*sizes)-1);
  for (int layer = 0; layer < weights.size(); layer++){
    weights[layer].resize(sizes[layer]);
    for (int sublayer = 0; sublayer < weights[layer].size(); sublayer++){
      weights[layer][sublayer].resize(sizes[layer+1]);
      for (int item = 0; item < weights[layer][sublayer].size(); item++){
	weights[layer][sublayer][item] = dist(randf);
      }
    }
  }
  //create biases
  vector<vector<float>> biases;
  biases.resize(sizeof(sizes)/sizeof(*sizes)-1);
  for (int layer = 0; layer < biases.size(); layer++){
    biases[layer].resize(sizes[layer+1]);
    for (int item = 0; item < biases[layer].size(); item++){
      biases[layer][item] = dist(randf);
    }
  }
  //create activations vector
  vector<vector<float>> activations;
  activations.resize(sizeof(sizes)/sizeof(*sizes));
  for (int layer = 0; layer < activations.size(); layer++){
    activations[layer].resize(sizes[layer]);
  }
  //create presigactivations vector
  vector<vector<float>> presigactivations;
  presigactivations.resize(sizeof(sizes)/sizeof(*sizes));
  for (int layer = 0; layer < presigactivations.size(); layer++){
    presigactivations[layer].reserve(largest);
    presigactivations[layer].resize(sizes[layer]);
  }

  //create variables used for storing data inbetween steps inside ff and backprop
  largest = 32;
  cout << "largest " << largest << endl;
  vector<float> ffx;
  ffx.resize(largest);
  vector<float> bpx;
  bpx.resize(sizes[2]);
  vector<float> bpy;
  bpy.resize(sizes[2]);
  vector<float> bpz;
  bpz.resize(sizes[2]);
  vector<float> bpsigprime;
  bpsigprime.resize(largest);
  vector<float> bpmatmulproduct;
  bpmatmulproduct.resize(largest);
  vector<vector<float>> transposedweights;
  transposedweights.resize(sizeof(sizes)/sizeof(*sizes));
  for (int layer = 0; layer < transposedweights.size(); layer++){
    transposedweights[layer].reserve(largest);
  }
  vector<vector<float>> delta;
  delta.resize(sizeof(sizes)/sizeof(*sizes)-1);
  for (int layer = 0; layer < delta.size(); layer++){
    delta[layer].resize(sizes[layer+1]);
  }
  vector<float> desiredoutput;
  desiredoutput.resize(sizes[2]);
  float cost;

  vector<vector<float>> images = loadimages();
  for (auto& image : images){
    for (auto& pixel : image){
      pixel = pixel / 255;
    }
  }
  vector<int> labels = loadlabels();
  bool nanfound = false;
  float eta = 1; //learning rate

  vector<vector<float>> nabla_b;
  nabla_b.resize(biases.size());
  vector<vector<vector<float>>> nabla_w;

  nabla_w.resize(weights.size());
  for (int l = 0; l < weights.size(); l++){
    nabla_w[l].resize(weights[l].size());
    for (int l2 = 0; l2 < weights[l].size(); l2++){
      nabla_w[l][l2].resize(weights[l][l2].size());
    }
  }


  for (int image = 0; image < 100; image++){
    cout << "=-=-=-=-=-=-=-=" << endl;
    activations[0] = images[image]; //set input activations
    desiredoutput = {0,0,0,0,0,0,0,0,0,0}; //set desired output
    desiredoutput[labels[image]] = 1;
    //run forwards pass
    feedforwards(weights, biases, activations, presigactivations, ffx);
    //run backwards pass
    //geterrors(weights, biases, activations, presigactivations, delta, desiredoutput, bpx, bpy, bpz, bpsigprime, bpmatmulproduct, transposedweights);
    //get cost (only required for tracking training)
    cost = MSE(activations.back(), desiredoutput);
    cout << "cost: " << cost << endl;

    //update biases
    for (int layer = 0; layer < biases.size(); layer++){
      nabla_b[layer] = delta[layer];
      for(auto& nbi : nabla_b[layer]){
	nbi = nbi * eta;
      }
      vectsub(biases[layer], nabla_b[layer], biases[layer]);
    }

  }
  cout << "activations:" << endl;
  printvectvect(activations);
  return 0;
}
