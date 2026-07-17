//! Tests for native grid pathfinding (`import pathfind`).

use data_code::{run, Value};

#[test]
fn pathfind_small_grid() {
    match run(
        r#"
import pathfind
path = pathfind.astar_grid(5, 5, (0, 0), (2, 2), set())
len(path)
"#,
    ) {
        Ok(Value::Number(n)) => assert!(n >= 5.0, "path too short: {}", n),
        Ok(v) => panic!("expected Number, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[test]
fn pathfind_blocked_unreachable() {
    match run(
        r#"
import pathfind
blocked = set()
blocked.add((0, 1))
pathfind.astar_grid(1, 3, (0, 0), (0, 2), blocked)
"#,
    ) {
        Ok(Value::Null) => {}
        Ok(v) => panic!("expected null, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[test]
fn pathfind_1000x5000_native() {
    match run(
        r#"
import pathfind
path = pathfind.astar_grid(1000, 5000, (0, 0), (559, 1234), set())
len(path)
"#,
    ) {
        Ok(Value::Number(n)) => {
            assert!(n >= 1794.0, "Manhattan min path length is 1793+1 nodes, got {}", n);
        }
        Ok(v) => panic!("expected Number, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}
