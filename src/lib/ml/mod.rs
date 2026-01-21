// ML module for DataCode

pub mod tensor;
pub mod autograd;
pub mod ops;
pub mod natives;

pub use autograd::{Variable, requires_grad};
pub mod graph;
pub mod model;
pub mod optimizer;
pub mod loss;
pub mod dataset;
pub mod layer;
pub mod device;
pub mod scheduler;
pub mod tensor_pool;
pub mod gpu_cache;
#[cfg(feature = "gpu")]
pub mod ops_gpu;

#[cfg(feature = "gpu")]
mod candle_integration;

pub mod backend_registry;

pub use backend_registry::BackendRegistry;
pub use tensor::Tensor;
pub use graph::{Graph, Node, NodeId, OpType};
pub use model::{LinearRegression, NeuralNetwork};
pub use optimizer::{SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW, OptimizerType};
pub use loss::{mse_loss, binary_cross_entropy_loss,
               mae_loss, huber_loss, hinge_loss, kl_divergence, smooth_l1_loss,
               categorical_cross_entropy_loss, sparse_softmax_cross_entropy_loss};
pub use dataset::Dataset;
pub use layer::{Layer, Linear, ReLU, Sigmoid, Tanh, Softmax, Flatten, Sequential, add_layer_to_registry, LayerId, with_layer, forward_layer_var};
pub use device::Device;
pub use tensor_pool::{TensorPool, get_tensor_from_pool, return_tensor_to_pool, clear_global_pool, get_pool_stats};
pub use gpu_cache::{GpuTensorCache, init_global_gpu_cache, get_gpu_tensor_from_cache, update_gpu_tensor_in_cache, clear_global_gpu_cache, get_gpu_cache_stats};
