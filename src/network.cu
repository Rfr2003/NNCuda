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
    if(cudaMalloc(dp, size) > 0) {
        printf("Malloc error for pointeur of size %d\n", size);
        exit(EXIT_FAILURE);
    }
}

void handle_copy_of_network(network* net_to_copy, network* other, int direction) {
    
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
    l->x = NULL; 
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