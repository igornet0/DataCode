// Текущие значения данных модуля у импортёра (`m.X`, повторный `from m import X`).
// Фикстура: tests/import_fixtures/modcalls/live.dc.
//
// Баг: `import m` сохранял в глобал `m` копию пространства имён на момент импорта, а
// `from m import X` копировал значение. После `m.bump()` выражение `m.COUNT` оставалось 0,
// повторный `from m import COUNT` тоже давал 0, а список из `from m import ITEMS` был копией:
// `push` с одной стороны не был виден другой. Словари модуля через `import m` вообще теряли
// изменения, сделанные функциями модуля.
//
// Теперь поля `m` указывают на живые ячейки данных модуля, а `from m import X` берёт текущее
// значение; массивы, множества и словари разделяются (как списки в Python), числа и строки
// копируются (переназначение у импортёра не меняет модуль).

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run_with_base_path, Value};
    use std::path::PathBuf;

    fn run(source: &str) -> Value {
        let mut base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        base.push("tests");
        base.push("import_fixtures");
        base.push("modcalls");
        run_with_base_path(source, base.as_path()).unwrap_or_else(|e| panic!("error: {:?}", e))
    }

    fn num(source: &str) -> f64 {
        match run(source) {
            Value::Int(IntValue::Finite(i)) => i as f64,
            Value::Float(f) => f.as_raw_f64(),
            Value::Number(n) => n,
            v => panic!("expected number, got {:?}", v),
        }
    }

    #[test]
    fn attribute_sees_rebound_number() {
        assert_eq!(num("import live\nlive.bump()\nlive.bump()\nlive.COUNT"), 2.0);
    }

    #[test]
    fn attribute_sees_pushed_items() {
        assert_eq!(num("import live\nlive.add(\"a\")\nlive.add(\"b\")\nlen(live.ITEMS)"), 3.0);
    }

    #[test]
    fn attribute_sees_rebound_list() {
        assert_eq!(num("import live\nlive.add(\"a\")\nlive.reset_items()\nlen(live.ITEMS)"), 0.0);
    }

    #[test]
    fn module_dict_mutation_persists_via_namespace() {
        assert_eq!(
            num("import live\nlive.set_cfg(\"a\", 1)\nlive.set_cfg(\"b\", 2)\nlen(live.CFG)"),
            3.0
        );
    }

    #[test]
    fn repeated_from_import_sees_current_value() {
        assert_eq!(
            num("from live import bump, COUNT\nbump()\nfrom live import COUNT\nCOUNT"),
            1.0
        );
    }

    #[test]
    fn from_import_number_is_value_at_import_time() {
        assert_eq!(num("from live import bump, COUNT\nbump()\nCOUNT"), 0.0);
    }

    #[test]
    fn from_import_list_is_shared_with_module() {
        assert_eq!(
            num("from live import add, ITEMS\nadd(\"a\")\nlen(ITEMS)"),
            2.0
        );
        assert_eq!(
            num("from live import items_len, ITEMS\npush(ITEMS, \"host\")\nitems_len()"),
            2.0
        );
    }

    #[test]
    fn from_import_dict_is_shared_with_module() {
        assert_eq!(
            num("from live import cfg_get, CFG\nCFG[\"n\"] = 5\ncfg_get(\"n\")"),
            5.0
        );
    }

    #[test]
    fn rebinding_imported_name_does_not_change_module() {
        assert_eq!(
            num("from live import ITEMS, items_len\nITEMS = [1, 2, 3, 4, 5]\nitems_len()"),
            1.0
        );
        assert_eq!(num("from live import COUNT, count\nCOUNT = 99\ncount()"), 0.0);
    }

    #[test]
    fn namespace_after_from_import_is_live() {
        assert_eq!(
            num("from live import bump\nbump()\nimport live\nlive.COUNT"),
            1.0
        );
    }

    #[test]
    fn caller_dict_with_same_name_is_isolated() {
        assert_eq!(
            num("CFG = {\"x\": 1, \"y\": 2, \"z\": 3, \"w\": 4}\nfrom live import set_cfg\nset_cfg(\"a\", 1)\nlen(CFG)"),
            4.0
        );
    }
}
