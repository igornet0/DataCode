// Autograd system for ML module
// Rewritten to match MetalNN architecture
// GradOp enum avoids Box<dyn Fn> in hot path (static dispatch in backward).

use crate::ml::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Gradient operation tag + captured data for backward. Static dispatch instead of Box<dyn Fn>.
#[derive(Clone)]
pub enum GradOp {
    Add {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    Mul {
        a_data: Tensor,
        b_data: Tensor,
    },
    Matmul {
        a_data: Tensor,
        b_data: Tensor,
    },
    Transpose,
    Relu {
        input_data: Tensor,
    },
}

/// Legacy type for ABI/forward compatibility where a closure is still used externally.
pub type GradientFn = Box<dyn Fn(&Tensor) -> Vec<(usize, Tensor)>>;

/// Узел в computational graph
pub struct Variable {
    pub data: Rc<RefCell<Tensor>>,
    pub grad: Rc<RefCell<Option<Tensor>>>,
    /// Static op for backward; avoids RefCell<Option<Box<dyn Fn>>> in hot path.
    pub grad_op: RefCell<Option<GradOp>>,
    #[allow(dead_code)]
    pub grad_fn: RefCell<Option<GradientFn>>,
    pub parents: Vec<Rc<Variable>>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub id: usize,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn get_next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field("id", &self.id)
            .finish()
    }
}

impl Variable {
    pub fn new(data: Tensor, requires_grad: bool) -> Rc<Self> {
        Rc::new(Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            grad_op: RefCell::new(None),
            grad_fn: RefCell::new(None),
            parents: vec![],
            requires_grad,
            is_leaf: true,
            id: get_next_id(),
        })
    }

    pub fn with_grad_fn(
        data: Tensor,
        requires_grad: bool,
        parents: Vec<Rc<Variable>>,
        grad_fn: GradientFn,
    ) -> Rc<Self> {
        Rc::new(Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            grad_op: RefCell::new(None),
            grad_fn: RefCell::new(Some(grad_fn)),
            parents,
            requires_grad,
            is_leaf: false,
            id: get_next_id(),
        })
    }

    /// Build variable with static GradOp (no Box<dyn Fn> in backward path).
    fn with_grad_op(
        data: Tensor,
        requires_grad: bool,
        parents: Vec<Rc<Variable>>,
        grad_op: GradOp,
    ) -> Rc<Self> {
        Rc::new(Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            grad_op: RefCell::new(Some(grad_op)),
            grad_fn: RefCell::new(None),
            parents,
            requires_grad,
            is_leaf: false,
            id: get_next_id(),
        })
    }

    pub fn backward(&self, grad: Tensor) {
        if !self.requires_grad {
            return;
        }

        // Обновляем градиент
        {
            let mut current_grad = self.grad.borrow_mut();
            match *current_grad {
                Some(ref existing_grad) => {
                    let sum = add_grads(existing_grad, &grad);
                    *current_grad = Some(sum);
                }
                None => {
                    *current_grad = Some(grad.clone());
                }
            }
        }

        // Backward: static dispatch via GradOp first, then fallback to legacy grad_fn
        let grad_ref = self.grad.borrow();
        let g = match grad_ref.as_ref() {
            Some(gg) => gg,
            None => return,
        };

        if let Some(ref op) = *self.grad_op.borrow() {
            let parent_grads = grad_op_backward(op, g);
            drop(grad_ref);
            for (idx, parent_grad) in parent_grads {
                if idx < self.parents.len() {
                    self.parents[idx].backward(parent_grad);
                }
            }
            return;
        }

        if let Some(ref f) = *self.grad_fn.borrow() {
            let parent_grads = f(g);
            drop(grad_ref);
            for (idx, parent_grad) in parent_grads {
                if idx < self.parents.len() {
                    self.parents[idx].backward(parent_grad);
                }
            }
        }
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }
}

fn add_grads(a: &Tensor, b: &Tensor) -> Tensor {
    use crate::ml::ops::add;
    add(a, b)
}

/// Static backward dispatch for GradOp (no dyn call in hot path).
fn grad_op_backward(op: &GradOp, grad: &Tensor) -> Vec<(usize, Tensor)> {
    use crate::ml::ops;
    match op {
        GradOp::Add { a_shape, b_shape } => {
            let grad_shape = grad.shape();
            let a_shape: &[usize] = a_shape;
            let b_shape: &[usize] = b_shape;
            if grad_shape == a_shape && grad_shape == b_shape {
                return vec![(0, grad.clone()), (1, grad.clone())];
            }
            let a_grad = if grad_shape == a_shape {
                grad.clone()
            } else {
                grad.clone()
            };
            let b_grad = if grad_shape == b_shape {
                grad.clone()
            } else if b_shape.len() == 1 && grad_shape.len() == 2 && b_shape[0] == grad_shape[1] {
                let grad_arr = grad.data();
                let mut summed = vec![0.0; b_shape[0]];
                for i in 0..grad_shape[0] {
                    for j in 0..b_shape[0] {
                        summed[j] += grad_arr[[i, j]];
                    }
                }
                Tensor::from_slice(&summed, b_shape)
            } else {
                grad.clone()
            };
            vec![(0, a_grad), (1, b_grad)]
        }
        GradOp::Mul { a_data, b_data } => {
            let a_grad = ops::mul(grad, b_data);
            let b_grad = ops::mul(grad, a_data);
            vec![(0, a_grad), (1, b_grad)]
        }
        GradOp::Matmul { a_data, b_data } => {
            let b_t = transpose(b_data);
            let a_t = transpose(a_data);
            let a_grad = ops::matmul(grad, &b_t);
            let b_grad = ops::matmul(&a_t, grad);
            vec![(0, a_grad), (1, b_grad)]
        }
        GradOp::Transpose => {
            let transposed_grad = transpose(grad);
            vec![(0, transposed_grad)]
        }
        GradOp::Relu { input_data } => {
            let arr = input_data.data();
            let mask = arr.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let mask_tensor = Tensor::from_array(mask);
            let input_grad = ops::mul(grad, &mask_tensor);
            vec![(0, input_grad)]
        }
    }
}

/// Создать Variable с requires_grad=true
pub fn requires_grad(t: Tensor) -> Rc<Variable> {
    Variable::new(t, true)
}

/// Операции с autograd

use crate::ml::ops;

/// Сложение с autograd
pub fn add_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    let a_data_ref = a.data.borrow();
    let b_data_ref = b.data.borrow();
    let result_data = ops::add(&a_data_ref, &b_data_ref);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let a_shape = a_data_ref.shape().to_vec();
    let b_shape = b_data_ref.shape().to_vec();
    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let grad_op = GradOp::Add {
        a_shape,
        b_shape,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// Умножение с autograd
pub fn mul_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    let a_data_ref = a.data.borrow();
    let b_data_ref = b.data.borrow();
    let result_data = ops::mul(&a_data_ref, &b_data_ref);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let grad_op = GradOp::Mul {
        a_data: a_data_ref.clone(),
        b_data: b_data_ref.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// Матричное умножение с autograd
pub fn matmul_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    let a_cpu = a.data.borrow().to_cpu()
        .map_err(|e| format!("Failed to convert tensor a to CPU: {}", e))
        .unwrap_or_else(|_| a.data.borrow().clone());
    let b_cpu = b.data.borrow().to_cpu()
        .map_err(|e| format!("Failed to convert tensor b to CPU: {}", e))
        .unwrap_or_else(|_| b.data.borrow().clone());

    let result_data = ops::matmul(&a_cpu, &b_cpu);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let grad_op = GradOp::Matmul {
        a_data: a_cpu.clone(),
        b_data: b_cpu.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

fn transpose(t: &Tensor) -> Tensor {
    // Convert to CPU if tensor is on GPU (ops require CPU tensors)
    let t_cpu = match t.to_cpu() {
        Ok(cpu_t) => cpu_t,
        Err(_e) => t.clone(),
    };
    
    // Validate tensor shape before transpose
    let shape = t_cpu.shape();
    if shape.len() != 2 {
        // For non-2D tensors, return as-is (transpose only works for 2D)
        return t_cpu;
    }
    
    // Use ops::transpose which handles 2D tensors correctly
    // This ensures proper handling of the tensor data
    crate::ml::ops::transpose(&t_cpu)
}

/// Транспонирование с autograd
pub fn transpose_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = transpose(&input_data_ref);

    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Transpose;
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// ReLU с autograd
pub fn relu_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::relu(&input_data_ref);

    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Relu {
        input_data: input_data_ref.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

