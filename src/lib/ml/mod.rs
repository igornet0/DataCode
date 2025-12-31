// ML module for DataCode

pub mod tensor;
pub mod natives;
pub mod graph;
pub mod model;
pub mod optimizer;
pub mod loss;
pub mod dataset;
pub mod layer;
pub mod device;
pub mod scheduler;
#[cfg(feature = "gpu")]
pub mod ops_gpu;

pub use tensor::Tensor;
pub use graph::{Graph, Node, NodeId, OpType};
pub use model::{LinearRegression, NeuralNetwork};
pub use optimizer::{SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW, OptimizerType};
pub use loss::{mse_loss, binary_cross_entropy_loss,
               mae_loss, huber_loss, hinge_loss, kl_divergence, smooth_l1_loss,
               categorical_cross_entropy_loss, sparse_softmax_cross_entropy_loss};
pub use dataset::Dataset;
pub use layer::{Layer, Linear, ReLU, Sigmoid, Tanh, Softmax, Flatten, Sequential, add_layer_to_registry};
pub use device::Device;

