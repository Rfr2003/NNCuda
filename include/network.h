#ifndef _NETWORK_
#define _NETWORK_ 

#define BATCH_SIZE 2

struct layer {
    int n; //taille layer sortie a
    int p; // taille layer entrée x
    float* w; // dim (n, p)
    float* x; // dim (p, 1) * BATCH_size
    float* b; // dim (n, 1)
    float* a; // dim (n, 1) * BATCH_size
    float* z; //dim (n, 1) * BATCH_size
    float* wT; //dim (p, n)
    float* aT; //dim (1, n) * BATCH_size
    float* da; //dim (n, 1) * BATCH_size
    float* dw; //dim (n, p) * BATCH_size
}typedef layer;

struct network {
    int nb_layers;
    layer** layers;
    float* y;
    float* error;
}typedef network;

float normal_distribution(float mean, float stddev);
void initialize_weights_he(float* w, int p, int n);
float random_float_0_to_1();
void handle_malloc(void** dp, int size);
void handle_copy_of_network(network* net_to_copy, network* other, int direction);
layer* create_layer(int p, int n);
network* create_network_with_layers(int nb_layers, ...);
network* create_empty_network();
void load_new_batch(float* x, float* y, network* net);
void add_layer_to_network(network* n, layer* l);
void layer_copy(layer* l, layer* dl, layer* l2);
void net_copy(network* net, network* dnet, network* net2);

#endif