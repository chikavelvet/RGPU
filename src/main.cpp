/*
 * File:   main.cu
 * Author: treyr3
 *
 * Created on April 17, 2018, 6:42 PM
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <atomic>
#include <random>
#include <time.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <iomanip>
#include <stack>
#include <map>
//#include <cuda_runtime.h>

using namespace std;

#define STREQ(s1, s2) (!strcmp(s1, s2))
#define SQR(x) ((x)*(x))

char* INPUT;
bool DEBUG, SERIAL;

/*
int CLUSTERS, ITERATIONS, WORKERS, STEP, THREADS_PER_BLOCK;
float THRESHOLD;
char* INPUT;
bool DEBUG, TIME_ONLY;

struct Cluster;
struct Point;

float* points;

__global__ void d_findNearestClusters (float *dataset, float *clusters, int *pcmap, int num_points, int num_clusters, int dim) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < num_points) {
	// For each point

	int nearestCluster = -1;
	float nearestSQDistance;

	// For each cluster
	for (int j = 0; j < num_clusters; ++j) {
	    float sqdist = 0;
	    for (int k = 0; k < dim; ++k) {
		sqdist += SQR(dataset[index * dim + k] - clusters[j * dim + k]);
	    }
	    if (nearestCluster < 0 || sqdist < nearestSQDistance) {
		nearestSQDistance = sqdist;
		nearestCluster = j;
	    }
	}

	pcmap[index] = nearestCluster;
    }
}

__global__ void d_findNearestClusters_s (float *dataset, float *clusters, int *pcmap, int num_points, int num_clusters, int dim, int block_size) {
    extern __shared__ int share[];
    int *nearestCluster = share;
    float *nearestSQDistance = (float*)&share[block_size];
    float *sqdist = (float*)&nearestSQDistance[block_size];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x;

    if (gindex < num_points) {
	// For each point

	nearestCluster[lindex] = -1;

	// For each cluster
	for (int j = 0; j < num_clusters; ++j) {
	    sqdist[lindex] = 0;
	    for (int k = 0; k < dim; ++k) {
		sqdist[lindex] += SQR(dataset[gindex * dim + k] - clusters[j * dim + k]);
	    }
	    if (nearestCluster[lindex] < 0 || sqdist[lindex] < nearestSQDistance[lindex]) {
		nearestSQDistance[lindex] = sqdist[lindex];
		nearestCluster[lindex] = j;
	    }
	}

	pcmap[gindex] = nearestCluster[lindex];
    }
}

__global__ void d_init_cpsize (int *cpsize, int num_clusters) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_clusters)
	cpsize[index] = 0;
}

__global__ void d_setup_cpsize (int *cpsize, int *pcmap, int num_points) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_points)
	cpsize[pcmap[index]] += 1;
}

struct KMeans {
    int num_clusters, num_points, dim, iterations;
    float* dataset;
    float* clusters;
    int* point_cluster_map;
    int* cluster_point_size;

    KMeans(float* ds, int _num_points, int k, int _dim) :
	dataset(ds),
	num_clusters(k),
	num_points(_num_points),
	dim(_dim)
    {
	clusters = (float*)malloc(sizeof(float) * num_clusters * dim);
	point_cluster_map = (int*)malloc(sizeof(int) * num_points);
	cluster_point_size = (int*)malloc(sizeof(int) * num_clusters);
    }

    void randomClusters () {
	for (int i = 0; i < num_clusters; ++i)
	    for (int j = 0; j < dim; ++j)
		clusters[i * dim + j] = dataset[(i % num_points) * dim + j];
    }

    void updateClusterCenters () {
	// Clear clusters
	for (int i = 0; i < num_clusters; ++i)
	    for (int j = 0; j < dim; ++j)
		clusters[i * dim + j] = 0.0;

	// For all points, add their values to the corresponding cluster
	for (int p = 0; p < num_points; ++p)
	    for (int j = 0; j < dim; ++j)
		clusters[point_cluster_map[p] * dim + j] += dataset[p * dim + j] / cluster_point_size[point_cluster_map[p]];
    }

    bool converged (float** old_center_values) {
	float maxdist = -1;
	for (int i = 0; i < num_clusters; ++i) {
	    float dist = 0;
	    for (int j = 0; j < dim; ++j) {
		dist += SQR(clusters[i * dim + j] - old_center_values[i][j]);
	    }
	    dist = sqrt(dist);
	    if (dist > maxdist)
		maxdist = dist;
	}
	return maxdist <= THRESHOLD;
    }

    void findNearestClusters () {
	if (STEP == 1) {
	    for (int index = 0; index < num_points; ++index) {
		// For each point

		int nearestCluster = -1;
		float nearestSQDistance;

		// For each cluster
		for (int j = 0; j < num_clusters; ++j) {
		    float sqdist = 0;
		    for (int k = 0; k < dim; ++k) {
			sqdist += SQR(dataset[index * dim + k] - clusters[j * dim + k]);
		    }
		    if (nearestCluster < 0 || sqdist < nearestSQDistance) {
			nearestSQDistance = sqdist;
			nearestCluster = j;
		    }
		}

		point_cluster_map[index] = nearestCluster;
	    }


	} else {
	    float *d_dataset, *d_clusters;
	    int *d_pcmap;//, *d_cpsize;

	    cudaMalloc((void **)&d_dataset, num_points * dim * sizeof(float));
	    cudaMalloc((void **)&d_clusters, num_clusters * dim * sizeof(float));
	    cudaMalloc((void **)&d_pcmap, num_points * sizeof(int));
	    //	    cudaMalloc((void **)&d_cpsize, num_clusters * sizeof(int));

	    cudaMemcpy(d_dataset, dataset, num_points * dim * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_clusters, clusters, num_clusters * dim * sizeof(float), cudaMemcpyHostToDevice);

	    if (STEP == 2)
		d_findNearestClusters<<<(num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_dataset, d_clusters, d_pcmap, num_points, num_clusters, dim);
	    else
		d_findNearestClusters_s<<<(num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK,
		    THREADS_PER_BLOCK*sizeof(int) + 2*THREADS_PER_BLOCK*sizeof(float)>>>(d_dataset, d_clusters, d_pcmap, num_points, num_clusters, dim, THREADS_PER_BLOCK);

	    cudaMemcpy(point_cluster_map, d_pcmap, num_points * sizeof(int), cudaMemcpyDeviceToHost);

	    cudaFree(d_dataset); cudaFree(d_clusters);
	    cudaFree(d_pcmap); //cudaFree(d_cpsize);
	}

	for (int j = 0; j < num_clusters; ++j)
	    cluster_point_size[j] = 0;

	for (int i = 0; i < num_points; ++i)
	    cluster_point_size[point_cluster_map[i]] += 1;
    }

    bool run () {
	// initialize centroids randomly
	randomClusters();

	// book-keeping
	iterations = 0;
	float** old_center_values;
	old_center_values = (float**)malloc(sizeof(float*) * num_clusters);

	for (int i = 0; i < num_clusters; ++i)
	    old_center_values[i] = (float*)malloc(sizeof(float) * dim);

	// core algorithm
	bool done = false;

	/*
	struct timespec start, finish;
	int elapsed_sec;
	double elapsed_ns;

	clock_gettime(CLOCK_MONOTONIC, &start);
	/

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	while (!done) {
	    ++iterations;

	    for (int i = 0; i < num_clusters; ++i)
		for (int j = 0; j < dim; ++j)
		    old_center_values[i][j] = clusters[i * dim + j];


	    this->findNearestClusters();

	    updateClusterCenters();

	    if (DEBUG) {
		cout << "After Iteration " << iterations << ":" << endl;
	    }

	    done = (ITERATIONS != 0 && iterations >= ITERATIONS) || converged(old_center_values);
	}
	cudaEventRecord(stop);

	/*
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_sec = static_cast<int>(finish.tv_sec - start.tv_sec);
	elapsed_ns = (finish.tv_nsec - start.tv_nsec);
	/

	cudaEventSynchronize(stop);
	float elapsed_msec = 0;
	cudaEventElapsedTime(&elapsed_msec, start, stop);

	cout << "Converged in " << iterations << " iterations (max=" << ITERATIONS << ")" << endl;
	cout << "parallel work completed in " << elapsed_msec << " msec" << endl;

	if (!TIME_ONLY)
	    for (int i = 0; i < num_clusters; ++i) {
		cout << "Cluster " << i << " center: ";
		for (int j = 0; j < dim; ++j)
		    cout << "[" << fixed << setprecision(3) << clusters[i * dim + j] << "]";
		cout << endl;
	    }

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return true;
    }
};
*/

const string type_str[] = { "operator", "number", "vector", "identifier", "delimiter" };
enum struct Type { OPERATOR, NUMBER, VECTOR, IDENTIFIER, DELIMITER };
const string which_str[] = { "+", "-", "*", "/", "=", "[", "]", "(", ")", "," };
enum struct Which { PLUS, MINUS, TIMES, DIVIDE, ASSIGN, LBRACKET, RBRACKET, LPAREN, RPAREN, COMMA };
/*map<Which, int> operator_precedence;
operator_precedence[Which::ASSIGN] = 0;
operator_precedence[Which::PLUS] = 1; operator_precedence[Which::MINUS] = 1;
operator_precedence[Which::TIMES] = 2; operator_precedence[Which::DIVIDE] = 2;
operator_precedence[Which::LBRACKET] = 10;*/

struct Token {
    Type toktype;
    int intval;
	int* vecval;
	Which whichval;
    string stringval;
    Token* operand;
    Token* link;
};

map<string, Token*> symbol_table;

vector<Token*> tokenize (const vector<string>& delimitedLine) {
    vector<Token*> tokenized(delimitedLine.size());

    for (int i = 0; i < delimitedLine.size(); ++i) {
      	Token* tok = new Token;
      	if (isdigit(delimitedLine[i][0])) {
      	    tok->toktype = Type::NUMBER;
      	    tok->intval = stoi(delimitedLine[i]);
      	} else if (delimitedLine[i] == "+") {
      	    tok->toktype = Type::OPERATOR;
      	    tok->whichval = Which::PLUS;
      	} else if (delimitedLine[i] == "-") {
      	    tok->toktype = Type::OPERATOR;
      	    tok->whichval = Which::MINUS;
      	} else if (delimitedLine[i] == "*") {
      	    tok->toktype = Type::OPERATOR;
      	    tok->whichval = Which::TIMES;
      	} else if ((delimitedLine[i] == "/")) {
      	    tok->toktype = Type::OPERATOR;
      	    tok->whichval = Which::DIVIDE;
      	} else if ((delimitedLine[i] == "=")) {
      	    tok->toktype = Type::OPERATOR;
      	    tok->whichval = Which::ASSIGN;
      	} else if ((delimitedLine[i] == "[")) {
      	    tok->toktype = Type::DELIMITER;
      	    tok->whichval = Which::LBRACKET;
      	} else if ((delimitedLine[i] == "]")) {
      	    tok->toktype = Type::DELIMITER;
      	    tok->whichval = Which::RBRACKET;
      	} else if ((delimitedLine[i] == "(")) {
      	    tok->toktype = Type::DELIMITER;
      	    tok->whichval = Which::LPAREN;
      	} else if ((delimitedLine[i] == ")")) {
      	    tok->toktype = Type::DELIMITER;
      	    tok->whichval = Which::RPAREN;
      	} else if ((delimitedLine[i] == ",")) {
      	    tok->toktype = Type::DELIMITER;
      	    tok->whichval = Which::COMMA;
      	} else {
      	    tok->toktype = Type::IDENTIFIER;
      	    tok->stringval = delimitedLine[i];
      	}
      	tokenized[i] = tok;
    }

    // if (DEBUG)
  	//  	for (Token* tok : tokenized)
  	//  		cout << tok << " of type " << static_cast<int>(tok->toktype) << endl;

    return tokenized;
}

int operator_precedence(Token* tok) {
	switch (tok->whichval) {
	case Which::ASSIGN:
		return 0; break;
	case Which::PLUS:
	case Which::MINUS:
		return 1; break;
	case Which::TIMES:
	case Which::DIVIDE:
		return 2; break;
	case Which::LBRACKET:
		return 10; break;
	}
	return -1;
}

string printtok(Token* tok) {
	stringstream ss;

	if (tok == NULL) {
		return "NULL";
	}

	switch (tok->toktype) {
	case Type::OPERATOR:
		ss << "(" << which_str[static_cast<int>(tok->whichval)] << " " << printtok(tok->operand) << " " << printtok(tok->operand->link) << ")";
		break;
	case Type::DELIMITER:
		ss << which_str[static_cast<int>(tok->whichval)];
		break;
	case Type::IDENTIFIER:
		return tok->stringval;
	case Type::NUMBER:
		ss << tok->intval;
		break;
	case Type::VECTOR:
		/*Token * it = tok->operand;
		ss << "[" << it->intval;
		it = it->link;
		while (it != NULL) {
			ss << ", " << it->intval;
			it = it->link;
		}
		ss << "]";*/
		ss << "[" << tok->vecval[0];
		for (int i = 1; i < tok->intval; ++i)
			ss << ", " << tok->vecval[i];
		ss << "]";
		break;
	}
	return ss.str();
}

string printsymtab(void) {
	stringstream ss;
	for (auto entry : symbol_table) 
		ss << entry.first << ": " << printtok(entry.second) << endl;
	return ss.str();
}

Token* parse(const vector<Token*>& tokenizedLine) {
	stack<Token*> operators;
	stack<Token*> operands;

	//shift-reduce strategy
	int deb = 0;
	for (Token* t : tokenizedLine) {
		// cout << "Parsing token " << ++deb << " with type " << static_cast<int>(t->toktype) << " ";
		// if (t->toktype == Type::OPERATOR || t->toktype == Type::DELIMITER)
		//   	cout << "(which: " << static_cast<int>(t->whichval) << ")";
		// cout << endl;

		if (t->toktype != Type::OPERATOR && t->toktype != Type::DELIMITER) {
			operands.push(t);
		}
		else {
			if (operators.empty()) {
				operators.push(t);
				continue;
			}

			if (t->whichval == Which::COMMA)
				continue;

			if (t->whichval == Which::LBRACKET) {
				operands.push(t);
				continue;
			}

			if (t->whichval == Which::RBRACKET) {
				int len = 0;
				while (!operands.empty()) {
					// cout << "Reducing bracket, top: " << printtok(operands.top()) << endl;
					Token* rest = operands.top(); operands.pop();
					Token* first = operands.top();

					if (first->toktype == Type::DELIMITER && first->whichval == Which::LBRACKET) {
						first->toktype = Type::VECTOR;
						int* vecval = (int*)malloc(++len * sizeof(int));
						Token* it = rest;
						for (int i = 0; i < len; ++i) {
							vecval[i] = rest->intval;
							rest = rest->link;
						}
						first->vecval = vecval;
						first->intval = len;
						break;
					}
					
					first->link = rest;
					++len;
				}

				cout << "Length: " << len << endl;

				continue;
			}

			while (!operators.empty() && operator_precedence(t) <= operator_precedence(operators.top())) {
				Token* op = operators.top(); operators.pop();
				Token* lhs = operands.top(); operands.pop();
				Token* rhs = operands.top(); operands.pop();
				op->operand = rhs;
				rhs->link = lhs;
				operands.push(op);
			}

			operators.push(t);
		}
	}

	while (!operators.empty()) {
		Token* op = operators.top(); operators.pop();
		Token* lhs = operands.top(); operands.pop();
		Token* rhs = operands.top(); operands.pop();
		op->operand = rhs;
		rhs->link = lhs;
		operands.push(op);
	}
	
	/*cout << "Final: " << endl;
	if (!operands.empty())
		cout << "    " << printtok(operands.top()) << endl;
	else
		cout << "Uh oh" << endl;*/

    return operands.top();
}

inline int ssplus(int  lhs, int  rhs) { return lhs + rhs;  }
int* vsplus(int* lhs, int  rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL) 
		for (int i = 0; i < size; ++i) 
			ret[i] = lhs[i] + rhs;
	return ret;
}
#define svplus(lhs, rhs, size) (vsplus(rhs,lhs,size))
int* vvplus(int* lhs, int* rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs[i] + rhs[i];
	return ret;
}

inline int ssminus(int  lhs, int  rhs) { return lhs - rhs; }
int* vsminus(int* lhs, int  rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs[i] - rhs;
	return ret;
}
int* svminus(int  lhs, int* rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs - rhs[i];
	return ret;
}
int* vvminus(int* lhs, int* rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs[i] - rhs[i];
	return ret;
}

inline int sstimes(int  lhs, int  rhs) { return lhs * rhs; }
int* vstimes(int* lhs, int  rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs[i] * rhs;
	return ret;
}
#define svtimes(lhs, rhs, size) (vstimes(rhs,lhs,size))
int* vvtimes(int* lhs, int* rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs[i] * rhs[i];
	return ret;
}

inline int ssdivide(int  lhs, int  rhs) { return lhs / rhs; }
int* vsdivide(int* lhs, int  rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs[i] / rhs;
	return ret;
}
int* svdivide(int  lhs, int* rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs / rhs[i];
	return ret;
}
int* vvdivide(int* lhs, int* rhs, int size) {
	int* ret = (int*)malloc(size * sizeof(int));
	if (SERIAL)
		for (int i = 0; i < size; ++i)
			ret[i] = lhs[i] / rhs[i];
	return ret;
}


Token* eval_plus(Token* lhs, Token* rhs) {
	Token * tok = new Token;
	if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::NUMBER) {
		tok->toktype = Type::NUMBER;
		tok->intval = ssplus(lhs->intval, rhs->intval);
	} else {
		tok->toktype = Type::VECTOR;
		if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::VECTOR) {
			tok->intval = lhs->intval;
			tok->vecval = vvplus(lhs->vecval, rhs->vecval, lhs->intval);
		}
		else if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::NUMBER) {
			tok->intval = lhs->intval;
			tok->vecval = vsplus(lhs->vecval, rhs->intval, lhs->intval);
		}
		else if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::VECTOR) {
			tok->intval = rhs->intval;
			tok->vecval = svplus(lhs->intval, rhs->vecval, rhs->intval);
		}
	}
	return tok;
}

Token* eval_minus(Token* lhs, Token* rhs) {
	Token * tok = new Token;
	if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::NUMBER) {
		tok->toktype = Type::NUMBER;
		tok->intval = ssminus(lhs->intval, rhs->intval);
	}
	else {
		tok->toktype = Type::VECTOR;
		if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::VECTOR) {
			tok->intval = lhs->intval;
			tok->vecval = vvminus(lhs->vecval, rhs->vecval, lhs->intval);
		}
		else if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::NUMBER) {
			tok->intval = lhs->intval;
			tok->vecval = vsminus(lhs->vecval, rhs->intval, lhs->intval);
		}
		else if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::VECTOR) {
			tok->intval = rhs->intval;
			tok->vecval = svminus(lhs->intval, rhs->vecval, rhs->intval);
		}
	}
	return tok;
}

Token* eval_times(Token* lhs, Token* rhs) {
	Token * tok = new Token;
	if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::NUMBER) {
		tok->toktype = Type::NUMBER;
		tok->intval = sstimes(lhs->intval, rhs->intval);
	}
	else {
		tok->toktype = Type::VECTOR;
		if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::VECTOR) {
			tok->intval = lhs->intval;
			tok->vecval = vvtimes(lhs->vecval, rhs->vecval, lhs->intval);
		}
		else if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::NUMBER) {
			tok->intval = lhs->intval;
			tok->vecval = vstimes(lhs->vecval, rhs->intval, lhs->intval);
		}
		else if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::VECTOR) {
			tok->intval = rhs->intval;
			tok->vecval = svtimes(lhs->intval, rhs->vecval, rhs->intval);
		}
	}
	return tok;
}

Token* eval_divide(Token* lhs, Token* rhs) {
	Token * tok = new Token;
	if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::NUMBER) {
		tok->toktype = Type::NUMBER;
		tok->intval = ssdivide(lhs->intval, rhs->intval);
	}
	else {
		tok->toktype = Type::VECTOR;
		if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::VECTOR) {
			tok->intval = lhs->intval;
			tok->vecval = vvdivide(lhs->vecval, rhs->vecval, lhs->intval);
		}
		else if (lhs->toktype == Type::VECTOR && rhs->toktype == Type::NUMBER) {
			tok->intval = lhs->intval;
			tok->vecval = vsdivide(lhs->vecval, rhs->intval, lhs->intval);
		}
		else if (lhs->toktype == Type::NUMBER && rhs->toktype == Type::VECTOR) {
			tok->intval = rhs->intval;
			tok->vecval = svdivide(lhs->intval, rhs->vecval, rhs->intval);
		}
	}
	return tok;
}

Token* evaluate_expr_tree(Token* root);

Token* evaluate_arith_serial(Token* root) {
	switch (root->whichval) {
	case Which::PLUS:
		return eval_plus(evaluate_expr_tree(root->operand), evaluate_expr_tree(root->operand->link));
	case Which::MINUS:
		return eval_minus(evaluate_expr_tree(root->operand), evaluate_expr_tree(root->operand->link));
	case Which::TIMES:
		return eval_times(evaluate_expr_tree(root->operand), evaluate_expr_tree(root->operand->link));
	case Which::DIVIDE:
		return eval_divide(evaluate_expr_tree(root->operand), evaluate_expr_tree(root->operand->link));
	}

	return NULL;
}

Token* evaluate_expr_tree(Token* root) {
	if (root == NULL)
		return NULL;

	Token * tok = new Token;
	switch (root->toktype) {
	case Type::NUMBER:
	case Type::VECTOR:
		return root;
	case Type::IDENTIFIER:
		return symbol_table[root->stringval];
	case Type::OPERATOR:
		switch (root->whichval) {
		case Which::ASSIGN:
			symbol_table[root->operand->stringval] = evaluate_expr_tree(root->operand->link);
			break;
		case Which::PLUS:
			return evaluate_arith_serial(root);
		case Which::MINUS:
			tok->toktype = Type::NUMBER;
			tok->intval = evaluate_expr_tree(root->operand)->intval - evaluate_expr_tree(root->operand->link)->intval;
			return tok;
		case Which::TIMES:
			tok->toktype = Type::NUMBER;
			tok->intval = evaluate_expr_tree(root->operand)->intval * evaluate_expr_tree(root->operand->link)->intval;
			return tok;
		case Which::DIVIDE:
			tok->toktype = Type::NUMBER;
			tok->intval = evaluate_expr_tree(root->operand)->intval / evaluate_expr_tree(root->operand->link)->intval;
			return tok;
		}
		break;
	}
	delete(tok);
	return NULL;
}

void execute_instruction(Token* inst) {
	cout << "Executing instruction " << printtok(inst) << endl;
	if (evaluate_expr_tree(inst) != NULL)
		cerr << "WARNING: instruction evaluation result non-null" << endl;
	cout << "Symbol table: " << endl;
	cout << printsymtab();
}

int main(int argc, char** argv) {
    // Defaults
    /*
    ITERATIONS = 0;
    THRESHOLD = 0.0000001;
    INPUT = NULL;
    CLUSTERS = 0;
    WORKERS = 1;
    STEP = 3;
    THREADS_PER_BLOCK = 256;
    */

    for (int i = 1; i < argc; ++i) {
		if (STREQ(argv[i], "--input")) {
			INPUT = argv[++i];
		} else if (STREQ(argv[i], "--debug")) {
			DEBUG = true;
		} else if (STREQ(argv[i], "--serial")) {
			SERIAL = true;
		}
		else {
			cout << "Unknown parameter: " << argv[i] << endl;
		}
    }

    if (DEBUG)
      cout << "Input: " << INPUT << endl;

    srand(time(NULL));

    // Parse input
    ifstream input;
    streambuf* orig_cin = 0;

    if (INPUT != NULL) {
	input.open(INPUT, ifstream::in);
	orig_cin = cin.rdbuf(input.rdbuf());
	cin.tie(&cout);
    }

    vector<Token*> instructions;

    string line;
    Token* parsed;
    while(!cin.eof()) {
		getline(cin, line);

		if (line == "") continue;

		if (DEBUG)
			cout << "Parsing line: " << line << endl;

		istringstream ss{line};
		istream_iterator<string> begin{ss};
		istream_iterator<string> end{};

		vector<string> delimited{begin, end};

		vector<Token*> tokenized = tokenize(delimited);
		parsed = parse(tokenized);

		instructions.push_back(parsed);
    }

	if (DEBUG) {
		cout << "Parsed instructions:" << endl;
		for (Token* i : instructions)
			cout << "    " << printtok(i) << endl;
	}

	for (Token* inst : instructions)
		execute_instruction(inst);

	cout << "Done." << endl;
    return 0;
}
