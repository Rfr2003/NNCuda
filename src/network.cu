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

void layer_copy(layer* l, layer* dl, layer* l2) {
    float *dw, *dx, *db, *da, *dz, *dwT, *daT, *dda, *ddw;
    int size;

    l2->n = l->n;
    l2->p = l->p;

    size = sizeof(float)*l->n*l->p;
    handle_malloc((void**)&dw, size);
    cudaMemcpy(dw, l->w, size, cudaMemcpyHostToDevice);
    l2->w = dw;

    size = sizeof(float)*l->p*BATCH_SIZE;
    handle_malloc((void**)&dx, size);
    cudaMemcpy(dx, l->x, size, cudaMemcpyHostToDevice);
    l2->x = dx;

    size = sizeof(float)*l->n;
    handle_malloc((void**)&db, size);
    cudaMemcpy(db, l->b, size, cudaMemcpyHostToDevice);
    l2->b = db;

    size = sizeof(float)*l->n*BATCH_SIZE;
    handle_malloc((void**)&da, size);
    cudaMemcpy(da, l->a, size, cudaMemcpyHostToDevice);
    l2->a = da;

    size = sizeof(float)*l->n*BATCH_SIZE;
    handle_malloc((void**)&dz, size);
    cudaMemcpy(dz, l->z, size, cudaMemcpyHostToDevice);
    l2->z = dz;

    size = sizeof(float)*l->n*l->p;
    handle_malloc((void**)&dwT, size);
    cudaMemcpy(dwT, l->wT, size, cudaMemcpyHostToDevice);
    l2->wT = dwT;

    size = sizeof(float)*l->n*BATCH_SIZE;
    handle_malloc((void**)&daT, size);
    cudaMemcpy(daT, l->aT, size, cudaMemcpyHostToDevice);
    l2->aT = daT;

    size = sizeof(float)*l->n*BATCH_SIZE;
    handle_malloc((void**)&dda, size);
    cudaMemcpy(dda, l->da, size, cudaMemcpyHostToDevice);
    l2->da = dda;

    size = sizeof(float)*l->n*BATCH_SIZE*l->p;
    handle_malloc((void**)&ddw, size);
    cudaMemcpy(ddw, l->dw, size, cudaMemcpyHostToDevice);
    l2->dw = ddw;

    cudaMemcpy(dl, &(l2), sizeof(layer), cudaMemcpyHostToDevice);
}

/* struct network {
    int nb_layers;
    layer** layers;
    float* y;
    float* error;
}typedef network; */

void net_copy(network* net, network* dnet, network* net2) {

    net2->nb_layers = net->nb_layers;
    handle_malloc((void**)&dnet, sizeof(network));
    int size;

    layer** dlayers;
    float *dy, *derror;

    size = sizeof(float)*BATCH_SIZE*net->layers[net->nb_layers-1]->n;
    handle_malloc((void**)&dy, size);
    cudaMemcpy(dy, net->y, size, cudaMemcpyHostToDevice);
    net2->y = dy;

    size = sizeof(float);
    handle_malloc((void**)&derror, size);
    cudaMemcpy(derror, net->error, size, cudaMemcpyHostToDevice);
    net2->error = derror;

    layer **layers2 = (layer**)malloc(sizeof(net->nb_layers)*sizeof(layer)); 

    layer* l, *dl, *l2;
    for(int i=0; i<net->nb_layers; i++) {
        l = net->layers[i];

        handle_malloc((void**)&dl, sizeof(layer));
        l2 = (layer*)malloc(sizeof(layer));

        layer_copy(l, dl, l2);

        layers2[i] = dl;
    }

    handle_malloc((void**)&dlayers, sizeof(layer*)*net->nb_layers);

    cudaMemcpy(dlayers, layers2, sizeof(net->nb_layers)*sizeof(layer), cudaMemcpyHostToDevice);

    net2->layers = dlayers;

    cudaMemcpy(dnet, &(net2), sizeof(network), cudaMemcpyHostToDevice);

    free(layers2);

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