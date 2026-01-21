// Единый тип значений для VM

use std::path::PathBuf;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use crate::common::table::Table;
use crate::ml::tensor::Tensor;
use crate::ml::graph::Graph;
use crate::ml::model::{LinearRegression, NeuralNetwork};
use crate::ml::optimizer::{SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW};
use crate::ml::dataset::Dataset;
use crate::ml::layer::{Sequential, LayerId};
use crate::plot::{Image, Figure, Axis, PlotWindowHandle};

#[derive(Debug)]
pub enum Value {
    Number(f64),
    Bool(bool),
    String(String),
    Array(Rc<RefCell<Vec<Value>>>),
    Tuple(Rc<RefCell<Vec<Value>>>),
    Function(usize), // Индекс функции в массиве функций
    NativeFunction(usize), // Индекс нативной функции
    Path(PathBuf), // Путь к файлу или директории
    Table(Rc<RefCell<Table>>),
    Object(HashMap<String, Value>), // Словарь/объект: ключ-значение
    ColumnReference {
        table: Rc<RefCell<Table>>,
        column_name: String,
    },
    Tensor(Rc<RefCell<Tensor>>),
    Graph(Rc<RefCell<Graph>>),
    LinearRegression(Rc<RefCell<LinearRegression>>),
    SGD(Rc<RefCell<SGD>>),
    Momentum(Rc<RefCell<Momentum>>),
    NAG(Rc<RefCell<NAG>>),
    Adagrad(Rc<RefCell<Adagrad>>),
    RMSprop(Rc<RefCell<RMSprop>>),
    Adam(Rc<RefCell<Adam>>),
    AdamW(Rc<RefCell<AdamW>>),
    Dataset(Rc<RefCell<Dataset>>),
    NeuralNetwork(Rc<RefCell<NeuralNetwork>>),
    Sequential(Rc<RefCell<Sequential>>),
    Layer(LayerId),
    Window(PlotWindowHandle), // Runtime only holds WindowId - Window lives in GUI thread
    Image(Rc<RefCell<Image>>),
    Figure(Rc<RefCell<Figure>>),
    Axis(Rc<RefCell<Axis>>),
    Null,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => *a.borrow() == *b.borrow(),
            (Value::Tuple(a), Value::Tuple(b)) => *a.borrow() == *b.borrow(),
            (Value::Function(a), Value::Function(b)) => a == b,
            (Value::NativeFunction(a), Value::NativeFunction(b)) => a == b,
            (Value::Path(a), Value::Path(b)) => a == b,
            (Value::Table(a), Value::Table(b)) => *a.borrow() == *b.borrow(),
            (Value::Object(a), Value::Object(b)) => a == b,
            (Value::ColumnReference { table: a, column_name: col_a }, Value::ColumnReference { table: b, column_name: col_b }) => {
                Rc::ptr_eq(a, b) && col_a == col_b
            },
            (Value::Tensor(a), Value::Tensor(b)) => *a.borrow() == *b.borrow(),
            (Value::Graph(a), Value::Graph(b)) => Rc::ptr_eq(a, b),
            (Value::LinearRegression(a), Value::LinearRegression(b)) => Rc::ptr_eq(a, b),
            (Value::SGD(a), Value::SGD(b)) => Rc::ptr_eq(a, b),
            (Value::Momentum(a), Value::Momentum(b)) => Rc::ptr_eq(a, b),
            (Value::NAG(a), Value::NAG(b)) => Rc::ptr_eq(a, b),
            (Value::Adagrad(a), Value::Adagrad(b)) => Rc::ptr_eq(a, b),
            (Value::RMSprop(a), Value::RMSprop(b)) => Rc::ptr_eq(a, b),
            (Value::Adam(a), Value::Adam(b)) => Rc::ptr_eq(a, b),
            (Value::AdamW(a), Value::AdamW(b)) => Rc::ptr_eq(a, b),
            (Value::Dataset(a), Value::Dataset(b)) => Rc::ptr_eq(a, b),
            (Value::NeuralNetwork(a), Value::NeuralNetwork(b)) => Rc::ptr_eq(a, b),
            (Value::Sequential(a), Value::Sequential(b)) => Rc::ptr_eq(a, b),
            (Value::Layer(a), Value::Layer(b)) => a == b,
            (Value::Window(a), Value::Window(b)) => a.id == b.id,
            (Value::Image(a), Value::Image(b)) => Rc::ptr_eq(a, b),
            (Value::Figure(a), Value::Figure(b)) => Rc::ptr_eq(a, b),
            (Value::Axis(a), Value::Axis(b)) => Rc::ptr_eq(a, b),
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

impl Value {
    /// Проверяет, можно ли использовать это значение как ключ кэша
    /// (только простые типы: Number, Bool, String, Null)
    pub fn is_hashable(&self) -> bool {
        matches!(self, Value::Number(_) | Value::Bool(_) | Value::String(_) | Value::Null)
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(false) => false,
            Value::Number(n) => *n != 0.0,
            Value::String(s) => !s.is_empty(),  // Пустая строка = false
            Value::Array(arr) => !arr.borrow().is_empty(),
            Value::Tuple(tuple) => !tuple.borrow().is_empty(),
            Value::Path(p) => !p.as_os_str().is_empty(),  // Путь не пустой = true
            Value::Table(table) => table.borrow().len() > 0,  // Таблица не пустая = true
            Value::Object(map) => !map.is_empty(),  // Объект не пустой = true
            Value::ColumnReference { table, column_name } => {
                let table_ref = table.borrow();
                if let Some(column) = table_ref.get_column(column_name) {
                    !column.is_empty()
                } else {
                    false
                }
            },
            Value::Tensor(tensor) => !tensor.borrow().data.is_empty(),
            Value::Graph(graph) => !graph.borrow().nodes.is_empty(),
            Value::LinearRegression(_) => true,
            Value::SGD(_) => true,
            Value::Momentum(_) => true,
            Value::NAG(_) => true,
            Value::Adagrad(_) => true,
            Value::RMSprop(_) => true,
            Value::AdamW(_) => true,
            Value::Dataset(dataset) => dataset.borrow().batch_size() > 0,
            Value::NeuralNetwork(_) => true,
            Value::Sequential(_) => true,
            Value::Layer(_) => true,
            Value::Window(_) => true,
            Value::Image(_) => true,
            Value::Figure(_) => true,
            Value::Axis(_) => true,
            _ => true,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            Value::Bool(b) => format!("{}", b),
            Value::String(s) => s.clone(),
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                let elements: Vec<String> = arr_ref.iter().map(|v| v.to_string()).collect();
                format!("[{}]", elements.join(", "))
            }
            Value::Tuple(tuple) => {
                let tuple_ref = tuple.borrow();
                let elements: Vec<String> = tuple_ref.iter().map(|v| v.to_string()).collect();
                format!("({})", elements.join(", "))
            }
            Value::Function(_) => "<function>".to_string(),
            Value::NativeFunction(_) => "<native function>".to_string(),
            Value::Path(p) => {
                // В режиме --use-ve показываем относительные пути
                use crate::websocket::{get_use_ve, get_user_session_path};
                if get_use_ve() {
                    if let Some(session_path) = get_user_session_path() {
                        // Канонизируем оба пути для корректного сравнения
                        let canonical_session = session_path.canonicalize().ok().unwrap_or(session_path);
                        let canonical_path = p.canonicalize().ok().unwrap_or(p.clone());
                        
                        // Проверяем, начинается ли путь с пути сессии
                        if let Ok(stripped) = canonical_path.strip_prefix(&canonical_session) {
                            // Формируем относительный путь с префиксом ./
                            let relative = stripped.to_string_lossy().to_string();
                            if relative.is_empty() || relative == "." {
                                "./".to_string()
                            } else {
                                // Убираем начальные слеши и добавляем ./
                                let trimmed = relative.trim_start_matches(|c| c == '/' || c == '\\');
                                if trimmed.is_empty() {
                                    "./".to_string()
                                } else {
                                    format!("./{}", trimmed)
                                }
                            }
                        } else {
                            // Путь вне сессии - возвращаем как есть (не канонизированный для сохранения оригинального формата)
                            p.to_string_lossy().to_string()
                        }
                    } else {
                        // Нет пути сессии - возвращаем как есть
                        p.to_string_lossy().to_string()
                    }
                } else {
                    // Не режим --use-ve - возвращаем полный путь
                    p.to_string_lossy().to_string()
                }
            },
            Value::Table(table) => {
                let t = table.borrow();
                format!("<table: {} rows, {} columns>", t.len(), t.column_count())
            }
            Value::ColumnReference { table, column_name } => {
                let t = table.borrow();
                if let Some(column) = t.get_column(column_name) {
                    format!("<column: {}.{} ({} values)>", 
                        t.name.as_ref().map(|n| n.as_str()).unwrap_or("table"),
                        column_name,
                        column.len())
                } else {
                    format!("<column: {}.{} (not found)>",
                        t.name.as_ref().map(|n| n.as_str()).unwrap_or("table"),
                        column_name)
                }
            }
            Value::Object(map) => {
                let pairs: Vec<String> = map.iter()
                    .map(|(k, v)| format!("\"{}\": {}", k, v.to_string()))
                    .collect();
                format!("{{{}}}", pairs.join(", "))
            }
            Value::Tensor(tensor) => {
                let t = tensor.borrow();
                format!("<tensor: shape={:?}, size={}>", t.shape, t.data.len())
            }
            Value::Graph(graph) => {
                let g = graph.borrow();
                format!("<graph: {} nodes, {} inputs>", g.nodes.len(), g.input_nodes.len())
            }
            Value::LinearRegression(lr) => {
                let model = lr.borrow();
                format!("<linear_regression: weights={:?}, bias={:?}>", 
                    model.get_weights().shape, model.get_bias().shape)
            }
            Value::SGD(sgd) => {
                let opt = sgd.borrow();
                format!("<sgd: lr={}>", opt.lr)
            }
            Value::Momentum(momentum) => {
                let opt = momentum.borrow();
                format!("<momentum: lr={}, beta={}>", opt.learning_rate, opt.beta)
            }
            Value::NAG(nag) => {
                let opt = nag.borrow();
                format!("<nag: lr={}, beta={}>", opt.learning_rate, opt.beta)
            }
            Value::Adagrad(adagrad) => {
                let opt = adagrad.borrow();
                format!("<adagrad: lr={}, epsilon={}>", opt.learning_rate, opt.epsilon)
            }
            Value::RMSprop(rmsprop) => {
                let opt = rmsprop.borrow();
                format!("<rmsprop: lr={}, gamma={}, epsilon={}>", opt.learning_rate, opt.gamma, opt.epsilon)
            }
            Value::Adam(adam) => {
                let opt = adam.borrow();
                format!("<adam: lr={}, beta1={}, beta2={}>", opt.lr, opt.beta1, opt.beta2)
            }
            Value::AdamW(adamw) => {
                let opt = adamw.borrow();
                format!("<adamw: lr={}, beta1={}, beta2={}, weight_decay={}>", opt.learning_rate, opt.beta1, opt.beta2, opt.weight_decay)
            }
            Value::Dataset(dataset) => {
                let d = dataset.borrow();
                format!("<dataset: batch_size={}, features={}, targets={}>", 
                    d.batch_size(), d.num_features(), d.num_targets())
            }
            Value::NeuralNetwork(_) => {
                format!("<neural_network>")
            }
            Value::Sequential(_) => {
                format!("<sequential>")
            }
            Value::Layer(id) => {
                format!("<layer: id={}>", id)
            }
            Value::Window(handle) => {
                format!("<window: id={:?}>", handle.id)
            }
            Value::Image(image) => {
                let img = image.borrow();
                format!("<image: {}x{}>", img.width, img.height)
            }
            Value::Figure(figure) => {
                let fig = figure.borrow();
                format!("<figure: {}x{} axes, figsize=({}, {})>", 
                    fig.axes.len(), 
                    if !fig.axes.is_empty() { fig.axes[0].len() } else { 0 },
                    fig.figsize.0, fig.figsize.1)
            }
            Value::Axis(_) => {
                format!("<axis>")
            }
            Value::Null => "null".to_string(),
        }
    }
}

// Реализуем Hash только для простых типов
// Для сложных типов Hash не реализован - они не могут быть ключами кэша
impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Number(n) => {
                // Хешируем число как байты для точности
                state.write_u8(0); // Тег для Number
                state.write_u64(n.to_bits());
            }
            Value::Bool(b) => {
                state.write_u8(1); // Тег для Bool
                state.write_u8(if *b { 1 } else { 0 });
            }
            Value::String(s) => {
                state.write_u8(2); // Тег для String
                s.hash(state);
            }
            Value::Null => {
                state.write_u8(3); // Тег для Null
            }
            // Для остальных типов не реализуем Hash - они не могут быть ключами кэша
            _ => {
                panic!("Cannot hash complex types (Array, Tuple, Table, Object, Function, Path)");
            }
        }
    }
}

impl Eq for Value {}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Number(n) => Value::Number(*n),
            Value::Bool(b) => Value::Bool(*b),
            Value::String(s) => Value::String(s.clone()),
            Value::Array(arr) => {
                // Создаем глубокую копию массива, рекурсивно клонируя все элементы
                let arr_ref = arr.borrow();
                let cloned_vec: Vec<Value> = arr_ref.iter().map(|v| v.clone()).collect();
                Value::Array(Rc::new(RefCell::new(cloned_vec)))
            },
            Value::Tuple(tuple) => {
                // Создаем глубокую копию кортежа, рекурсивно клонируя все элементы
                let tuple_ref = tuple.borrow();
                let cloned_vec: Vec<Value> = tuple_ref.iter().map(|v| v.clone()).collect();
                Value::Tuple(Rc::new(RefCell::new(cloned_vec)))
            },
            Value::Function(idx) => Value::Function(*idx),
            Value::NativeFunction(idx) => Value::NativeFunction(*idx),
            Value::Path(p) => Value::Path(p.clone()),
            Value::Table(table) => {
                // Создаем новый Rc с глубокой копией таблицы
                Value::Table(Rc::new(RefCell::new(table.borrow().clone())))
            },
            Value::ColumnReference { table, column_name } => {
                // Для ColumnReference клонируем ссылку на таблицу и имя колонки
                Value::ColumnReference {
                    table: table.clone(),
                    column_name: column_name.clone(),
                }
            },
            Value::Object(map) => {
                // Создаем глубокую копию объекта (клонируем каждое значение)
                let mut cloned_map = HashMap::new();
                for (k, v) in map {
                    cloned_map.insert(k.clone(), v.clone());
                }
                Value::Object(cloned_map)
            },
            Value::Tensor(tensor) => {
                // Создаем новый Rc с глубокой копией тензора
                Value::Tensor(Rc::new(RefCell::new(tensor.borrow().clone())))
            },
            Value::Graph(graph) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Graph(graph.clone())
            },
            Value::LinearRegression(lr) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::LinearRegression(lr.clone())
            },
            Value::SGD(sgd) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::SGD(sgd.clone())
            },
            Value::Momentum(momentum) => {
                Value::Momentum(momentum.clone())
            },
            Value::NAG(nag) => {
                Value::NAG(nag.clone())
            },
            Value::Adagrad(adagrad) => {
                Value::Adagrad(adagrad.clone())
            },
            Value::RMSprop(rmsprop) => {
                Value::RMSprop(rmsprop.clone())
            },
            Value::Adam(adam) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Adam(adam.clone())
            },
            Value::AdamW(adamw) => {
                Value::AdamW(adamw.clone())
            },
            Value::Dataset(dataset) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Dataset(dataset.clone())
            },
            Value::NeuralNetwork(nn) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::NeuralNetwork(nn.clone())
            },
            Value::Sequential(seq) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Sequential(seq.clone())
            },
            Value::Layer(id) => Value::Layer(*id),
            Value::Window(handle) => {
                // WindowHandle is Copy, so just copy it
                Value::Window(*handle)
            },
            Value::Image(image) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Image(image.clone())
            },
            Value::Figure(figure) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Figure(figure.clone())
            },
            Value::Axis(axis) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Axis(axis.clone())
            },
            Value::Null => Value::Null,
        }
    }
}

