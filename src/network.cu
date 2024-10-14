#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

#include "network.h"

float normal_distribution(float mean, float stddev) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stddev + mean;
}

// Initialisation des poids selon la méthode de He
void initialize_weights_he(float* w, int p, int n) {
    // Calcul de l'écart type (stddev)
    float stddev = sqrt(2.0 / p);

    // Remplir les poids avec des valeurs suivant la distribution normale
    for (int i=0; i<n; i++) {
        for(int j=0; j<p; j++) {
            w[i*p + j] = normal_distribution(0.0, stddev);
        }
    }
}

float random_float_0_to_1() {
    return (float)rand() / (float)RAND_MAX;
}

void handle_malloc(void** dp, int size) {
    int error = cudaMalloc(dp, size);
    if(error > 0) {
        printf("Malloc error %d for pointeur of size %d\n", error, size);
        exit(EXIT_FAILURE);
    }
}

/* int n; //taille layer sortie a
    int p; // taille layer entrée x
    float* w; // dim (n, p)
    float* x; // dim (p, 1) * BATCH_size
    float* b; // dim (n, 1)
    float* a; // dim (n, 1) * BATCH_size
    float* z; //dim (n, 1) * BATCH_size
    float* wT; //dim (p, n)
    float* aT; //dim (1, n) * BATCH_size
    float* da; //dim (n, 1) * BATCH_size
    float* dw; //dim (n, p) * BATCH_size */

void layer_copy_HostToDevice(layer* l, float *dw, float *dx, float *db, float *da, float *dz, float *dwT, float *daT, float *dda, float *ddw) {
    
    int size;

    size = sizeof(float) * l->n * l->p;
    cudaMemcpy(dw, l->w, size, cudaMemcpyHostToDevice);

    size = sizeof(float) * l->p * BATCH_SIZE;
    cudaMemcpy(dx, l->x, size, cudaMemcpyHostToDevice);

    size = sizeof(float) * l->n;
    cudaMemcpy(db, l->b, size, cudaMemcpyHostToDevice);

    size = sizeof(float) * l->n * BATCH_SIZE;
    cudaMemcpy(da, l->a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, l->z, size, cudaMemcpyHostToDevice);
    
    size = sizeof(float) * l->n * l->p;
    cudaMemcpy(dwT, l->wT, size, cudaMemcpyHostToDevice);

    size = sizeof(float) * l->n * BATCH_SIZE;
    cudaMemcpy(daT, l->aT, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dda, l->da, size, cudaMemcpyHostToDevice);

    size = sizeof(float) * l->n * BATCH_SIZE * l->p;
    cudaMemcpy(ddw, l->dw, size, cudaMemcpyHostToDevice);

}

void layer_copy_DeviceToHost(layer* l, float *dw, float *dx, float *db, float *da, float *dz, float *dwT, float *daT, float *dda, float *ddw) {
    int size;

    size = sizeof(float)*l->n*l->p;
    cudaMemcpy(l->w, dw, size, cudaMemcpyDeviceToHost);

    size = sizeof(float)*l->p*BATCH_SIZE;
    cudaMemcpy(l->x, dx, size, cudaMemcpyDeviceToHost);


    size = sizeof(float)*l->n;
    cudaMemcpy(l->b, db, size, cudaMemcpyDeviceToHost);

    size = sizeof(float)*l->n*BATCH_SIZE;
    cudaMemcpy(l->a, da, size, cudaMemcpyDeviceToHost);

    size = sizeof(float)*l->n*BATCH_SIZE;
    cudaMemcpy(l->z, dz, size, cudaMemcpyDeviceToHost);

    size = sizeof(float)*l->n*l->p;
    cudaMemcpy(l->wT, dwT, size, cudaMemcpyDeviceToHost);

    size = sizeof(float)*l->n*BATCH_SIZE;
    cudaMemcpy(l->aT, daT, size, cudaMemcpyDeviceToHost);

    size = sizeof(float)*l->n*BATCH_SIZE;
    cudaMemcpy(l->da, dda, size, cudaMemcpyDeviceToHost);

    size = sizeof(float)*l->n*BATCH_SIZE*l->p;
    cudaMemcpy(l->dw, ddw, size, cudaMemcpyDeviceToHost);
}

//initialise les connexions entre deux layers
layer* create_layer(int p, int n) {

    if(p > 1024 || n > 1024) {
        printf("Layers with dimensionality above 1024 are prohibited\n");
        exit(EXIT_FAILURE);
    }
    layer* l = (layer*)malloc(sizeof(layer));

    l->n = n;
    l->p = p;

    l->w = (float*)malloc(sizeof(float)*n*p); 
    l->x = (float*)malloc(sizeof(float)*p*BATCH_SIZE); 
    l->b = (float*)malloc(sizeof(float)*n);
    l->a = (float*)malloc(sizeof(float)*n*BATCH_SIZE);
    l->z = (float*)malloc(sizeof(float)*n*BATCH_SIZE);

    l->wT = (float*)malloc(sizeof(float)*n*p*BATCH_SIZE);
    l->aT = (float*)malloc(sizeof(float)*n*BATCH_SIZE);

    //multiplier par batch size
    l->dw = (float*)malloc(sizeof(float)*n*p*BATCH_SIZE);
    l->da = (float*)malloc(sizeof(float)*n*BATCH_SIZE);

    initialize_weights_he(l->w, l->p, l->n);

    for(int i=0; i<l->n; i++) {
        l->b[i] = 0;
    }

    return l;
}



network* create_network_with_layers(int nb_layers, ...) {
    va_list args;

    network* n = (network*)malloc(sizeof(network));
    n->nb_layers = nb_layers;
    n->layers = (layer**)malloc(sizeof(layer*)*nb_layers);

    va_start(args, nb_layers);

    for(int i=0; i<nb_layers; i++) {
        n->layers[i] = va_arg(args, layer*);
    }

    va_end(args);

    return n;
}

network* create_empty_network() {
    network* n = (network*)malloc(sizeof(network));
    n->layers = NULL;
    n->nb_layers = 0;
    n->error = (float*)malloc(sizeof(float));
    n->y = NULL;

    return n;
}

void load_new_batch(float* x, float* y, network* net) {
    if(net->nb_layers == 0) {
        printf("Empty network\n");
        exit(EXIT_FAILURE);
    }

    net->y = y;

    net->layers[0]->x = x;

    net->error[0] = 0;
}

void add_layer_to_network(network* n, layer* l) {
    if(n->nb_layers > 0) {
        if(n->layers[n->nb_layers-1]->n != l->p) {
            printf("Incompatible size for layers : size n for layer (i) must be equal t osize p for layer (i+1)\n");
            exit(EXIT_FAILURE);
        }
    }

    n->nb_layers += 1;

    if(n->layers == NULL) {
        n->layers = (layer**)malloc(sizeof(layer*));
    }
    else{
        n->layers = (layer**)realloc(n->layers, sizeof(layer*)*(n->nb_layers));
    }

    n->layers[n->nb_layers-1] = l;
}