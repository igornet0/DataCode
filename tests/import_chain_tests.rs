// Цепочка from-импортов: main -> game -> potions -> names (фикстуры: tests/import_fixtures/chain).
//
// Баг: при `from game import ...` всем функциям, добавленным из VM модуля `game`, ставился модуль
// `game`, включая функции `potions`, которые `game` сам импортировал. `names_count` из `potions`
// искала `NAMES` в пространстве имён `game`, где это имя есть только как пустой слот слияния, и
// получала `null`. Тот же сценарий через пакет с `__lib__.dc` (`from shop import Game`).

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run_with_base_path, Value};
    use std::path::PathBuf;

    fn num(source: &str) -> f64 {
        let mut base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        base.push("tests");
        base.push("import_fixtures");
        base.push("chain");
        match run_with_base_path(source, base.as_path()) {
            Ok(Value::Int(IntValue::Finite(i))) => i as f64,
            Ok(Value::Float(f)) => f.as_raw_f64(),
            Ok(Value::Number(n)) => n,
            other => panic!("expected number, got {:?}", other),
        }
    }

    #[test]
    fn second_level_function_via_first_level_function() {
        assert_eq!(num("from game import run_game\nrun_game()"), 32.0);
    }

    #[test]
    fn second_level_function_via_class_method() {
        assert_eq!(num("from game import Game\ng = Game()\ng.run()"), 32.0);
    }

    #[test]
    fn second_level_function_called_directly() {
        assert_eq!(num("from potions import names_count\nnames_count()"), 32.0);
    }

    #[test]
    fn package_with_lib_reexport() {
        assert_eq!(num("from shop import Game\ng = Game()\ng.run()"), 32.0);
    }
}
