// Autograd system for ML module
// Rewritten to match MetalNN architecture

use crate::ml::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// Градиент функция - принимает градиент и возвращает градиенты для родителей
pub type GradientFn = Box<dyn Fn(&Tensor) -> Vec<(usize, Tensor)>>;

/// Узел в computational graph
pub struct Variable {
    pub data: Rc<RefCell<Tensor>>,
    pub grad: Rc<RefCell<Option<Tensor>>>,
    #[allow(dead_code)]
    pub grad_fn: RefCell<Option<GradientFn>>,
    pub parents: Vec<Rc<Variable>>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub id: usize,
}

static mut NEXT_ID: usize = 0;

fn get_next_id() -> usize {
    unsafe {
        let id = NEXT_ID;
        NEXT_ID += 1;
        id
    }
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
            grad_fn: RefCell::new(Some(grad_fn)),
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
                    // Складываем градиенты (для узлов с несколькими исходящими рёбрами)
                    let sum = add_grads(existing_grad, &grad);
                    *current_grad = Some(sum);
                }
                None => {
                    *current_grad = Some(grad.clone());
                }
            }
        }

        // Вызываем backward для родителей
        let grad_fn = self.grad_fn.borrow();
        if let Some(ref grad_fn) = *grad_fn {
            let current_grad = self.grad.borrow();
            if let Some(ref g) = *current_grad {
                let parent_grads = grad_fn(g);
                for (idx, parent_grad) in parent_grads {
                    if idx < self.parents.len() {
                        self.parents[idx].backward(parent_grad);
                    }
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

/// Создать Variable с requires_grad=true
pub fn requires_grad(t: Tensor) -> Rc<Variable> {
    Variable::new(t, true)
}

/// Операции с autograd

use crate::ml::ops;

/// Сложение с autograd
pub fn add_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    // Используем ссылки на данные вместо клонирования
    let a_data_ref = a.data.borrow();
    let b_data_ref = b.data.borrow();
    let result_data = ops::add(&a_data_ref, &b_data_ref);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    // Сохраняем формы для правильной обработки broadcast в backward
    let a_shape = a_data_ref.shape().to_vec();
    let b_shape = b_data_ref.shape().to_vec();
    
    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let grad_fn: GradientFn = Box::new(move |grad| {
        let grad_shape = grad.shape();
        
        // Если формы совпадают, просто передаем градиент
        if grad_shape == a_shape.as_slice() && grad_shape == b_shape.as_slice() {
            return vec![(0, grad.clone()), (1, grad.clone())];
        }
        
        // Обработка broadcast: если b был broadcast (меньшая размерность),
        // градиент для b должен суммироваться по broadcast размерностям
        let a_grad = if grad_shape == a_shape.as_slice() {
            grad.clone()
        } else {
            // Если градиент имеет другую форму, пытаемся суммировать по лишним размерностям
            // Это упрощенная версия - в реальности нужна более сложная логика
            grad.clone()
        };
        
        let b_grad = if grad_shape == b_shape.as_slice() {
            grad.clone()
        } else if b_shape.len() == 1 && grad_shape.len() == 2 && b_shape[0] == grad_shape[1] {
            // b был [features], grad имеет [batch, features]
            // Суммируем по batch размерности
            let grad_arr = grad.data();
            let mut summed = vec![0.0; b_shape[0]];
            for i in 0..grad_shape[0] {
                for j in 0..b_shape[0] {
                    summed[j] += grad_arr[[i, j]];
                }
            }
            Tensor::from_slice(&summed, &b_shape)
        } else {
            grad.clone()
        };
        
        vec![(0, a_grad), (1, b_grad)]
    });

    Variable::with_grad_fn(result_data, requires_grad, parents, grad_fn)
}

/// Умножение с autograd
pub fn mul_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    // Используем ссылки на данные вместо клонирования для forward pass
    let a_data_ref = a.data.borrow();
    let b_data_ref = b.data.borrow();
    let result_data = ops::mul(&a_data_ref, &b_data_ref);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    // Для backward pass нужно клонировать данные, так как они используются в замыкании
    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let a_data_clone = a_data_ref.clone();
    let b_data_clone = b_data_ref.clone();
    let grad_fn: GradientFn = Box::new(move |grad| {
        // Градиент для умножения: d(x*y)/dx = y, d(x*y)/dy = x
        let a_grad = ops::mul(grad, &b_data_clone);
        let b_grad = ops::mul(grad, &a_data_clone);
        vec![(0, a_grad), (1, b_grad)]
    });

    Variable::with_grad_fn(result_data, requires_grad, parents, grad_fn)
}

/// Матричное умножение с autograd
pub fn matmul_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    // Convert to CPU if tensors are on GPU (ops::matmul requires CPU tensors)
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

    // Для backward pass нужно клонировать данные, так как они используются в замыкании
    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let a_data_clone = a_cpu.clone();
    let b_data_clone = b_cpu.clone();
    let grad_fn: GradientFn = Box::new(move |grad| {
        // Градиент для matmul:
        // dL/da = grad @ b^T
        // dL/db = a^T @ grad
        let b_t = transpose(&b_data_clone);
        let a_t = transpose(&a_data_clone);
        let a_grad = ops::matmul(grad, &b_t);
        let b_grad = ops::matmul(&a_t, grad);
        vec![(0, a_grad), (1, b_grad)]
    });

    Variable::with_grad_fn(result_data, requires_grad, parents, grad_fn)
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
    // Используем ссылку на данные вместо клонирования
    let input_data_ref = input.data.borrow();
    let result_data = transpose(&input_data_ref);

    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&input)];
    // Градиент транспозиции - это снова транспозиция
    let grad_fn: GradientFn = Box::new(move |grad| {
        let transposed_grad = transpose(grad);
        vec![(0, transposed_grad)]
    });

    Variable::with_grad_fn(result_data, requires_grad, parents, grad_fn)
}

/// ReLU с autograd
pub fn relu_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    // Используем ссылку на данные вместо клонирования для forward pass
    let input_data_ref = input.data.borrow();
    let result_data = ops::relu(&input_data_ref);

    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    // Для backward pass нужно клонировать данные для создания маски
    let parents = vec![Rc::clone(&input)];
    let input_data_clone = input_data_ref.clone();
    let grad_fn: GradientFn = Box::new(move |grad| {
        // Градиент ReLU: 1 если x > 0, иначе 0
        let arr = input_data_clone.data();
        let mask = arr.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let mask_tensor = Tensor::from_array(mask);
        let input_grad = ops::mul(grad, &mask_tensor);
        vec![(0, input_grad)]
    });

    Variable::with_grad_fn(result_data, requires_grad, parents, grad_fn)
}

