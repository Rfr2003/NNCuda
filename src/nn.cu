#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

#include "network.h"

#define OUTPUT 2 //taille output
#define LEARNING_RATE 0.1f


void read_csv(float* x, float* y, int* number_columns, int* number_observations, char* file_name) {
    FILE *file;
    char *buffer;
    long file_size;

    // Ouvrir le fichier en mode binaire
    file = fopen(file_name, "rb");
    if (file == NULL) {
        perror("Erreur lors de l'ouverture du fichier");
        exit(EXIT_FAILURE);
    }

    // Se positionner à la fin du fichier pour déterminer sa taille
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file); // Revenir au début du fichier

    // Allouer la mémoire pour stocker le contenu du fichier
    buffer = (char *)malloc(sizeof(char) * file_size + 1);
    if (buffer == NULL) {
        perror("Erreur d'allocation de mémoire");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Lire tout le fichier dans le buffer
    fread(buffer, sizeof(char), file_size, file);
    buffer[file_size] = '\0';

    //calcul nombre de colonnes et nombre de lignes
    *number_columns = 0;
    *number_observations = 0;

    int c;
    bool first_row = true;

    while((c = fgetc(file)) != EOF) {
        if(first_row && c == ',') {
            *number_columns += 1;
        }
        
        if(c == '\n') {
            *number_observations += 1;
        }
        
    }

    x = (float*)malloc(sizeof(float)*(*number_columns)*(*number_observations));
    y = (float*)malloc(sizeof(float)*(*number_observations)*3);

    int nb_char_per_line = (*number_columns)*30;

    char* line = (char*)malloc(sizeof(char)*nb_char_per_line); //chaque colonne ne dépasse pas 30 caractères
    char * strToken;
    int count;
    int line_count = 0;
    char *endptr;

    while(fgets(line, nb_char_per_line, file) != NULL) {
        count = 0;
        strToken = strtok (line, ",");

        while ( strToken != NULL ) {
            if(count < (*number_columns)) {
                x[count + line_count*(*number_columns)] = strtof(strToken, &endptr);
            }
            else{
                float num = strtof(strToken, &endptr);
                for(int i=0; i<3; i++) {
                    y[line_count*3 + i] = (num == (float)i) * i;
                }
            }
            strToken = strtok ( NULL, "," );
        }

        line_count++;
    }




    fclose(file);
    free(buffer);
}




void cross_entropy(network* net) {
    net->error[0] = 0;
    for(int i=0; i<net->layers[net->nb_layers-1]->n*BATCH_SIZE; i++) {
        net->error[0] += net->y[i] * logf(net->layers[net->nb_layers-1]->a[i]);
    }
    net->error[0] *= -1;
}

__device__ void k_transpose_w(float* w, float* wT, int n, int p) {
    int lid = threadIdx.x;
    int begin_batch = blockIdx.x * n * p;

    if(lid < n) {
        for(int j=0; j<p; j++) {
            wT[j*n + begin_batch + lid] = w[begin_batch + lid*p + j];
        }
    }
}

//calcule le gradient pour la dernière couche 
__device__ void k_gradient_last_layer(float* da, float* a, float* y, int n) {
    int lid = threadIdx.x;
    int ind = blockIdx.x*n + lid;

    if(lid < n) {
        da[ind] = a[ind] - y[ind];
    }
}

//calcule : grad_l {p_l+1/n_l, 1} = (w_l+1)^T{p_l+1, n_l+1} * grad_l+1 {n_l+1, 1} * deriv(ReLu(z_l) {n_l, 1})
// p = p_l+1 et n = n_l+1
__device__ void k_gradient_hidden_layer(float* da, float* da_next, float* w, float* z, int p, int n) {
    int lid = threadIdx.x;

    __shared__ float sh_da_next[1024];

    int begin_batch = blockIdx.x*p;
    float res = 0;

    if(lid < n) {
        sh_da_next[lid] = da_next[lid + blockIdx.x*n];
    }
    __syncthreads();


    if (lid < p){

        for(int row=0; row<n; row++) {
            res += w[lid*n + row] * sh_da_next[row];
        }

        da[lid + begin_batch] = res * (z[lid + begin_batch] > 0);

    }
}

//dw {n_l, p_l} = (grad_l) {n_l, 1} * (a_l-1) {1, p_l}
__device__ void k_gradient_w(float* dw, float* da, float* a_previous, int n, int p){
    int lid = threadIdx.x;

    __shared__ float shda[1024];
    __shared__ float sha[1024];

    int begin_batch_w = blockIdx.x * n * p;
    int begin_batch_da = blockIdx.x * n;
    int begin_batch_a = blockIdx.x * p;

    if(lid < n) {
        shda[lid] = da[begin_batch_da + lid];
    }

    if(lid < p) {
        sha[lid] = a_previous[begin_batch_a + lid];
    }

    __syncthreads();

    if(lid < n) {
        for(int col=0; col<p; col++) {
            dw[begin_batch_w + lid*n + col] = shda[lid] * sha[col];
        }
    }
}

// w_l {n_l, p_l} = w_l {n_l, p_l} - lr * dw {n_l, p_l}
__device__ void k_update_weights(float* w, float* dw, int n, int p, float learn_rate) {
    int lid = threadIdx.x;
    int begin_batch = blockIdx.x * n * p;

    if(n > p) {
        if(lid < n) {
            for(int col=0; col<p; col++) {
                atomicAdd(&w[lid*n + col],  -learn_rate * w[begin_batch + lid*n + col]/BATCH_SIZE);
            }
        }
    }
    else {
        if(lid < p) {
            for(int row=0; row<n; row++) {
                atomicAdd(&w[lid*p + row], -learn_rate * w[begin_batch + lid*p + row]/BATCH_SIZE);
            }
        }
    }
}

// b_l {n_l, 1} = b_l {n_l, 1} - lr * da{n_l, 1}
__device__ void k_update_bias(float* b, float* da, int n, float learn_rate) {
    int lid = threadIdx.x;
    int begin_batch = blockIdx.x * n;

    if (lid < n){    
        atomicAdd(&b[lid], -learn_rate * da[begin_batch + lid]/BATCH_SIZE);
    }
}


__global__ void k_back_propagation(network* net) {
    layer* l;
    for(int i=net->nb_layers-1; i>-1;i--) {
        l = net->layers[i];
        if(i == net->nb_layers-1) {
            k_gradient_last_layer(l->da, l->a, net->y, l->n);
        }
        else {
            k_transpose_w(net->layers[i+1]->w, net->layers[i+1]->wT, net->layers[i+1]->n, net->layers[i+1]->p);

            __syncthreads();

            k_gradient_hidden_layer(l->da, net->layers[i+1]->da, net->layers[i+1]->wT, l->z, net->layers[i+1]->p, net->layers[i+1]->n);
        }

        __syncthreads();

        if(i > 0) {
            k_gradient_w(l->dw, l->da, net->layers[i-1]->a, l->n, l->p);
        }
        else {
            k_gradient_w(l->dw, l->da, l->x, l->n, l->p);
        }

        __syncthreads();

        k_update_weights(l->w, l->dw, l->n, l->p, LEARNING_RATE);

        __syncthreads();

        k_update_bias(l->b, l->da, l->n, LEARNING_RATE);

        __syncthreads();
    }
}

// result{n, 1} = o(w_{n, p} * x_{p, 1} + b_{n, 1})
__device__ void k_feed_forward(float* x, float* w, float* bias, float* z, float* a, int n, int p, bool last_layer) {
    int lid = threadIdx.x;

    __shared__ float shx[1024];

    int begin_batch = blockIdx.x*n;
    float res = 0;

    if(lid < p) {
        shx[lid] = x[lid + blockIdx.x*p];
    }
    __syncthreads();


    if (lid < n){


        for(int col=0; col<p; col++) {
            res += w[lid*p + col] * shx[col];
        }

        res += bias[lid + begin_batch];

        z[lid + begin_batch] = res;

    }

    __syncthreads();

    if(lid < n) {
        //activation softmax pour le dernier layer
        if(last_layer) {
            float sum = 0;
            for(int j=0; j<n; j++) {
                sum += expf(z[j + begin_batch]);
            }
            a[lid + begin_batch] = expf(z[lid + begin_batch])/sum; 
        }
        //activation ReLu pour les autres
        else {
            a[lid + begin_batch] = z[lid + begin_batch] * (z[lid + begin_batch] >= 0);
        }
    }
}

__global__ void k_step(network* n) {
    bool last_layer;
    for(int i=0; i<n->nb_layers; i++) {
        layer* l = n->layers[i];

        last_layer = i == (n->nb_layers-1);

        if (i == 0) {
            k_feed_forward(l->x, l->w, l->b, l->z, l->a, l->n, l->p, last_layer);
        }
        else
        {
            k_feed_forward(n->layers[i-1]->a, l->w, l->b, l->z, l->a, l->n, l->p, last_layer);
        }
    } 
}

int main(int argc, char **argv){

    network* net = create_empty_network();

    add_layer_to_network(net, create_layer(2, 4));
    add_layer_to_network(net, create_layer(4, 2));

    float x[4] = {1, 1, 1, 1};
    float y[2*BATCH_SIZE] = {1.0f, 0, 1.0f, 0}; //nombre de classes * nombre de d'obervastions

    load_new_batch(x, y, net);

    network *dnet;
    handle_malloc((void**)&dnet, sizeof(network));

    cudaMemcpy(dnet, net, sizeof(network), cudaMemcpyHostToDevice);

    printf("b² before update: \n");
    for(int i=0; i<net->layers[1]->n; i++){
        printf("%f\n", net->layers[1]->b[i]);
    }

    printf("W² before update: \n");
    for(int i=0; i<net->layers[1]->n*net->layers[1]->p; i++){
        printf("%f\n", net->layers[1]->w[i]);
    } 

    k_step<<<BATCH_SIZE, 1024>>>(dnet);

    k_back_propagation<<<BATCH_SIZE, 1024>>>(dnet);

    cudaMemcpy(net, dnet, sizeof(network), cudaMemcpyDeviceToHost);

    printf("X : \n");
    for(int i=0; i<net->layers[0]->p*BATCH_SIZE; i++){
        if(i%2 == 0) {
            printf("----------%d\n", i/2);
        }
        printf("%f\n", net->layers[0]->x[i]);
    }

    printf("W¹ : \n");
    for(int i=0; i<net->layers[0]->n*net->layers[0]->p; i++){
        if(i%net->layers[0]->p == 0) {
            printf("----------%d\n", i/net->layers[0]->p);
        }
        printf("%f\n", net->layers[0]->w[i]);
    }

    printf("Z¹ : \n");
    for(int i=0; i<net->layers[0]->n*BATCH_SIZE; i++){
        if(i%2 == 0) {
            printf("----------%d\n", i/2);
        }
        printf("%f\n", net->layers[0]->z[i]);
    }

    printf("A¹ : \n");
    for(int i=0; i<net->layers[0]->n*BATCH_SIZE; i++){
        if(i%2 == 0) {
            printf("----------%d\n", i/2);
        }
        printf("%f\n", net->layers[0]->a[i]);
    }

    printf("b² after update: %d\n", net->layers[1]->n);
    for(int i=0; i<net->layers[1]->n; i++){
        printf("%f\n", net->layers[1]->b[i]);
    }

    printf("W² after update: \n");
    for(int i=0; i<net->layers[1]->n*net->layers[1]->p; i++){
        if(i%net->layers[1]->p == 0) {
            printf("----------%d\n", i/net->layers[1]->p);
        }
        printf("%f\n", net->layers[1]->w[i]);
    }

    printf("Z² : \n");
    for(int i=0; i<net->layers[1]->n*BATCH_SIZE; i++){
        if(i%2 == 0) {
            printf("----------%d\n", i/2);
        }
        printf("%f\n", net->layers[1]->z[i]);
    }

    printf("results : \n");
    for(int i=0; i<net->layers[net->nb_layers-1]->n*BATCH_SIZE; i++){
        if(i%2 == 0) {
            printf("----------%d\n", i/2);
        }
        printf("%f\n", net->layers[net->nb_layers-1]->a[i]);
    }

    printf("WT : \n");
    for(int i=0; i<net->layers[1]->n*net->layers[0]->p; i++){
        printf("%f\n", net->layers[1]->wT[i]);
    }

    printf("Error : %f\n", net->error[0]);

    cross_entropy(net);

    printf("Error : %f\n", net->error[0]);
    
    cudaFree(dnet);

    return 0;
}