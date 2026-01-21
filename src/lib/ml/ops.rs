// Basic operations for ML module
// Rewritten to match MetalNN architecture

use crate::ml::tensor::Tensor;

/// Сложение двух тензоров
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    
    // Проверяем, нужен ли broadcast
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Если формы одинаковые, просто складываем
    if a_shape == b_shape {
        let result = a_arr + b_arr;
        return Tensor::from_array(result);
    }
    
    // Если b - это 1D тензор, который нужно broadcast к a
    // Например: a = [batch, features], b = [features] -> broadcast b к [1, features]
    if b_shape.len() == 1 && a_shape.len() == 2 && b_shape[0] == a_shape[1] {
        // Broadcast b: [features] -> [1, features] -> [batch, features]
        let b_2d = b_arr.clone().insert_axis(ndarray::Axis(0));
        let b_broadcast = b_2d.broadcast(a_shape).expect("Failed to broadcast").to_owned();
        let result = a_arr + &b_broadcast;
        return Tensor::from_array(result);
    }
    
    // Пробуем автоматический broadcast через ndarray
    let result = a_arr + b_arr;
    Tensor::from_array(result)
}

/// Умножение двух тензоров (элементное)
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    let result = a_arr * b_arr;
    Tensor::from_array(result)
}

/// Скалярное умножение
pub fn scalar_mul(a: &Tensor, scalar: f32) -> Tensor {
    let a_arr = a.data();
    let result = a_arr * scalar;
    Tensor::from_array(result)
}

/// Деление двух тензоров (элементное)
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    let result = a_arr / b_arr;
    Tensor::from_array(result)
}

/// Деление тензора на скаляр
pub fn scalar_div(a: &Tensor, scalar: f32) -> Tensor {
    let a_arr = a.data();
    let result = a_arr / scalar;
    Tensor::from_array(result)
}

/// Квадратный корень (элементный)
pub fn sqrt(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| x.sqrt());
    Tensor::from_array(result)
}

/// Матричное умножение (оптимизированная версия)
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    
    // Оптимизированная реализация для 2D тензоров
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let (_m, k) = (a_shape[0], a_shape[1]);
        let (k2, _n) = (b_shape[0], b_shape[1]);
        
        assert_eq!(k, k2, "Matrix dimensions don't match for multiplication");
        
        // Используем Array2 для эффективных операций
        // Используем view для избежания копирования данных
        let a_2d = a_arr
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert to 2D array");
        let b_2d = b_arr
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert to 2D array");
        
        // Используем встроенное матричное умножение ndarray (оптимизированное)
        // to_owned() создает копию только результата, а не входных данных
        let result = a_2d.dot(&b_2d);
        
        // Преобразуем обратно в ArrayD
        let result_d = result.into_dyn();
        Tensor::from_array(result_d)
    } else {
        panic!("matmul currently supports only 2D tensors");
    }
}

/// Транспонирование
pub fn transpose(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.t().to_owned();
    Tensor::from_array(result)
}

/// ReLU активация
pub fn relu(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| if x > 0.0 { x } else { 0.0 });
    Tensor::from_array(result)
}

/// Sigmoid активация
pub fn sigmoid(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    Tensor::from_array(result)
}

/// Tanh активация
pub fn tanh(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| x.tanh());
    Tensor::from_array(result)
}

/// Softmax (правильная реализация по оси)
pub fn softmax(t: &Tensor, axis: usize) -> Tensor {
    let arr = t.data();
    let shape = t.shape();
    
    // Для 2D тензора [batch, classes] с axis=1
    if shape.len() == 2 && axis == 1 {
        let batch_size = shape[0];
        let num_classes = shape[1];
        let mut result_data = Vec::with_capacity(batch_size * num_classes);
        
        for i in 0..batch_size {
            // Находим максимум по классам для этого образца
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                max_val = max_val.max(arr[[i, j]]);
            }
            
            // Вычисляем exp(x - max) для численной стабильности
            let mut exp_sum = 0.0;
            let mut exp_vals = Vec::with_capacity(num_classes);
            for j in 0..num_classes {
                let exp_val = (arr[[i, j]] - max_val).exp();
                exp_sum += exp_val;
                exp_vals.push(exp_val);
            }
            
            // Нормализуем
            for exp_val in exp_vals {
                result_data.push(exp_val / exp_sum);
            }
        }
        
        Tensor::from_slice(&result_data, shape)
    } else {
        // Fallback для других случаев - вычисляем по всему тензору
        let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_arr = arr.mapv(|x| (x - max_val).exp());
        let sum: f32 = exp_arr.sum();
        let result = exp_arr.mapv(|x| x / sum);
        Tensor::from_array(result)
    }
}

/// Sum to shape (для broadcast градиентов)
pub fn sum_to_shape(t: &Tensor, target_shape: &[usize]) -> Result<Tensor, String> {
    let arr = t.data();
    let current_shape = t.shape();
    
    // Если формы уже совпадают, возвращаем как есть
    if current_shape == target_shape {
        return Ok(t.clone());
    }
    
    // Упрощенная реализация: reshape если возможно, иначе суммируем
    let total_elements: usize = target_shape.iter().product();
    if arr.len() == total_elements {
        // Просто reshape
        let reshaped = arr.clone().into_shape(ndarray::IxDyn(target_shape)).unwrap().to_owned();
        Ok(Tensor::from_array(reshaped))
    } else {
        // Нужно суммировать - упрощенная версия
        let mut result = arr.clone();
        // Суммируем по всем осям кроме последней
        while result.shape().len() > target_shape.len() {
            result = result.sum_axis(ndarray::Axis(0));
        }
        // Reshape к целевой форме
        if result.len() == total_elements {
            let reshaped = result.into_shape(ndarray::IxDyn(target_shape)).unwrap().to_owned();
            Ok(Tensor::from_array(reshaped))
        } else {
            Err(format!("Cannot sum from {:?} to {:?}", current_shape, target_shape))
        }
    }
}

