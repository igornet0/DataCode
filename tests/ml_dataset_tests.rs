// Tests for ML Dataset from Tables

use data_code::{run, Value};

#[test]
fn test_dataset_from_table() {
    // Test creating dataset from table
    let code = r#"
        import ml
        let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        let headers = ["x1", "x2", "y"]
        let my_table = table(data, headers)
        
        let feature_cols = ["x1", "x2"]
        let target_cols = ["y"]
        let ds = ml.dataset(my_table, feature_cols, target_cols)
        
        typeof(ds)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "dataset", "Expected dataset type, got {}", s);
        }
        _ => panic!("Expected String for type"),
    }
}

#[test]
fn test_dataset_features() {
    // Test extracting features from dataset
    let code = r#"
        import ml
        let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let headers = ["x1", "x2", "y"]
        let my_table = table(data, headers)
        
        let feature_cols = ["x1", "x2"]
        let target_cols = ["y"]
        let ds = ml.dataset(my_table, feature_cols, target_cols)
        
        let features = ml.dataset_features(ds)
        ml.shape(features)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Features shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 2, "Batch size should be 2");
                    assert_eq!(*n2 as usize, 2, "Feature count should be 2");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_dataset_targets() {
    // Test extracting targets from dataset
    let code = r#"
        import ml
        let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let headers = ["x1", "x2", "y"]
        let my_table = table(data, headers)
        
        let feature_cols = ["x1", "x2"]
        let target_cols = ["y"]
        let ds = ml.dataset(my_table, feature_cols, target_cols)
        
        let targets = ml.dataset_targets(ds)
        ml.shape(targets)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Targets shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 2, "Batch size should be 2");
                    assert_eq!(*n2 as usize, 1, "Target count should be 1");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_dataset_with_linear_regression() {
    // Test using dataset with linear regression
    let code = r#"
        import ml
        let data = [[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 4.0, 7.0]]
        let headers = ["x1", "x2", "y"]
        let my_table = table(data, headers)
        
        let feature_cols = ["x1", "x2"]
        let target_cols = ["y"]
        let ds = ml.dataset(my_table, feature_cols, target_cols)
        
        let features = ml.dataset_features(ds)
        let targets = ml.dataset_targets(ds)
        
        let model = ml.linear_regression(2)
        let loss_history = ml.lr_train(model, features, targets, 10, 0.01)
        
        len(loss_history)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 10, "Loss history should have 10 entries");
        }
        _ => panic!("Expected Number for loss history length"),
    }
}

#[test]
fn test_dataset_multiple_targets() {
    // Test dataset with multiple target columns
    let code = r#"
        import ml
        let data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        let headers = ["x1", "x2", "y1", "y2"]
        let my_table = table(data, headers)
        
        let feature_cols = ["x1", "x2"]
        let target_cols = ["y1", "y2"]
        let ds = ml.dataset(my_table, feature_cols, target_cols)
        
        let targets = ml.dataset_targets(ds)
        let shape = ml.shape(targets)
        shape[1]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 2, "Should have 2 target columns");
        }
        _ => panic!("Expected Number for target count"),
    }
}

#[test]
fn test_load_mnist_train() {
    // Test loading MNIST train dataset
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        typeof(dataset_train)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "dataset", "Expected dataset type, got {}", s);
        }
        _ => panic!("Expected String for type"),
    }
}

#[test]
fn test_load_mnist_test() {
    // Test loading MNIST test dataset
    let code = r#"
        import ml
        let dataset_test = ml.load_mnist("test")
        typeof(dataset_test)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "dataset", "Expected dataset type, got {}", s);
        }
        _ => panic!("Expected String for type"),
    }
}

#[test]
fn test_mnist_dataset_size() {
    // Test MNIST dataset batch size
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        len(dataset_train)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            // MNIST train should have 60000 samples
            assert!(n > 0.0, "Dataset should have samples");
            assert_eq!(n as usize, 60000, "MNIST train should have 60000 samples");
        }
        _ => panic!("Expected Number for dataset size"),
    }
}

#[test]
fn test_mnist_dataset_features_shape() {
    // Test MNIST dataset features shape
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let features = ml.dataset_features(dataset_train)
        ml.shape(features)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Features shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 60000, "Batch size should be 60000");
                    assert_eq!(*n2 as usize, 784, "Feature count should be 784 (28x28)");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_mnist_dataset_targets_shape() {
    // Test MNIST dataset targets shape
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let targets = ml.dataset_targets(dataset_train)
        ml.shape(targets)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Targets shape should have 2 dimensions");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(n1), Value::Number(n2)) => {
                    assert_eq!(*n1 as usize, 60000, "Batch size should be 60000");
                    assert_eq!(*n2 as usize, 1, "Target count should be 1");
                }
                _ => panic!("Expected Number values in shape"),
            }
        }
        _ => panic!("Expected Array for shape"),
    }
}

#[test]
fn test_mnist_dataset_element_access() {
    // Test accessing dataset elements by index
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let sample = dataset_train[0]
        len(sample)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 2, "Sample should be [features, target]");
        }
        _ => panic!("Expected Number for sample length"),
    }
}

#[test]
fn test_mnist_dataset_iteration() {
    // Test iterating over MNIST dataset
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let count = 0
        for x, y in dataset_train {
            count = count + 1
            if count >= 10 {
                break
            }
        }
        count
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 10, "Should iterate 10 times");
        }
        _ => panic!("Expected Number for count"),
    }
}

#[test]
fn test_mnist_dataset_label_values() {
    // Test that labels are valid (0-9)
    let code = r#"
        import ml
        let dataset_train = ml.load_mnist("train")
        let labels = []
        let count = 0
        for x, y in dataset_train {
            labels = labels + [y]
            count = count + 1
            if count >= 100 {
                break
            }
        }
        let min_label = 10.0
        let max_label = -1.0
        for label in labels {
            if label < min_label {
                min_label = label
            }
            if label > max_label {
                max_label = label
            }
        }
        [min_label, max_label]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 2, "Should return [min, max]");
            match (&arr_ref[0], &arr_ref[1]) {
                (Value::Number(min), Value::Number(max)) => {
                    assert!(*min >= 0.0 && *min <= 9.0, "Min label should be 0-9");
                    assert!(*max >= 0.0 && *max <= 9.0, "Max label should be 0-9");
                }
                _ => panic!("Expected Number values for min/max"),
            }
        }
        _ => panic!("Expected Array for min/max labels"),
    }
}

