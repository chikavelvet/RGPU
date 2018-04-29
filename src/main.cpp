/* 
 * File:   main.cu
 * Author: treyr3
 *
 * Created on April 17, 2018, 6:42 PM
 */

#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <string>
#include <cstdint>
#include <vector>
#include <atomic>
#include <sys/sysinfo.h>
#include <random>
#include <time.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <iomanip>
//#include <cuda_runtime.h>

using namespace std;

#define STREQ(s1, s2) (!strcmp(s1, s2))
#define SQR(x) ((x)*(x))

char* INPUT;
bool DEBUG;

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

enum struct Type { OPERATOR, NUMBER, IDENTIFIER, DELIMITER };
enum struct Which { PLUS, MINUS, TIMES, DIVIDE };

struct Token {
    Type toktype;
    union value {
	int intval;
	Which whichval;
	string stringval;
    };
};

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
	} else {
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

    string line;
    vector<string> delimited;
    Token parsedLine;
    while(!cin.eof()) {
	getline(cin, line);

	cout << line << endl;
    }
    
    return 0;
}

