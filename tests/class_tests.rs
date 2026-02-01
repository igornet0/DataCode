// Тесты для классов DataCode
// Покрывают: парсинг cls, конструкторы, методы, this, поля (private/public)

#[cfg(test)]
mod tests {
    use data_code::{run, Value};
    use data_code::parser::{Parser, Stmt};
    use data_code::lexer::Lexer;

    fn parse(source: &str) -> Vec<Stmt> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        parser.parse().unwrap()
    }

    fn run_ok(source: &str) -> Value {
        run(source).expect("expected Ok")
    }

    fn assert_number(source: &str, expected: f64) {
        let v = run_ok(source);
        if let Value::Number(n) = v {
            assert_eq!(n, expected, "expected {}, got {}", expected, n);
        } else {
            panic!("expected Number({}), got {:?}", expected, v);
        }
    }

    fn assert_null(source: &str) {
        let v = run_ok(source);
        assert!(matches!(v, Value::Null), "expected Null, got {:?}", v);
    }

    fn run_err(source: &str) -> Result<Value, data_code::common::error::LangError> {
        data_code::run(source)
    }

    fn assert_runtime_error(source: &str, substring: &str) {
        let result = run_err(source);
        assert!(result.is_err(), "expected Err, got Ok");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains(substring),
            "expected error containing {:?}, got: {}",
            substring,
            err_msg
        );
    }

    // ========== Парсер: объявление класса ==========

    #[test]
    fn test_parser_class_empty() {
        let source = r#"
            cls Empty {
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, private_fields, public_fields, private_variables, public_variables, constructors, methods, .. } = &stmts[0] {
            assert_eq!(name, "Empty");
            assert!(private_fields.is_empty());
            assert!(public_fields.is_empty());
            assert!(private_variables.is_empty());
            assert!(public_variables.is_empty());
            assert!(constructors.is_empty());
            assert!(methods.is_empty());
        } else {
            panic!("Expected Class statement, got {:?}", stmts[0]);
        }
    }

    #[test]
    fn test_parser_class_variable_without_public_goes_to_private() {
        let source = r#"
            cls Config {
                model_config = 42
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, private_variables, public_variables, .. } = &stmts[0] {
            assert_eq!(name, "Config");
            assert_eq!(private_variables.len(), 1);
            assert_eq!(private_variables[0].name, "model_config");
            assert!(public_variables.is_empty());
        } else {
            panic!("Expected Class with private variable");
        }
    }

    #[test]
    fn test_parser_class_variable_after_public_goes_to_public() {
        let source = r#"
            cls C {
                public:
                    x = 1
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, private_variables, public_variables, .. } = &stmts[0] {
            assert_eq!(name, "C");
            assert!(private_variables.is_empty());
            assert_eq!(public_variables.len(), 1);
            assert_eq!(public_variables[0].name, "x");
        } else {
            panic!("Expected Class with public variable");
        }
    }

    #[test]
    fn test_parser_class_mix_variable_and_fields() {
        let source = r#"
            cls Data {
                secret = 0
                public:
                    name: str
                    tag = 99
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, private_variables, public_fields, public_variables, .. } = &stmts[0] {
            assert_eq!(name, "Data");
            assert_eq!(private_variables.len(), 1);
            assert_eq!(private_variables[0].name, "secret");
            assert_eq!(public_fields.len(), 1);
            assert_eq!(public_fields[0].name, "name");
            assert_eq!(public_variables.len(), 1);
            assert_eq!(public_variables[0].name, "tag");
        } else {
            panic!("Expected Class with mix of variable and fields");
        }
    }

    #[test]
    fn test_parser_class_with_public_fields() {
        let source = r#"
            cls Point {
                public:
                    x: int
                    y: int
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, public_fields, .. } = &stmts[0] {
            assert_eq!(name, "Point");
            assert_eq!(public_fields.len(), 2);
            assert_eq!(public_fields[0].name, "x");
            assert_eq!(public_fields[1].name, "y");
        } else {
            panic!("Expected Class with public fields");
        }
    }

    #[test]
    fn test_parser_class_with_private_and_public() {
        let source = r#"
            cls Data {
                private:
                    id: int
                    balance: float = 0.0
                public:
                    name: str
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, private_fields, public_fields, .. } = &stmts[0] {
            assert_eq!(name, "Data");
            assert_eq!(private_fields.len(), 2);
            assert_eq!(private_fields[0].name, "id");
            assert_eq!(private_fields[1].name, "balance");
            assert!(private_fields[1].default_value.is_some());
            assert_eq!(public_fields.len(), 1);
            assert_eq!(public_fields[0].name, "name");
        } else {
            panic!("Expected Class with private and public fields");
        }
    }

    #[test]
    fn test_parser_class_with_protected_section() {
        // Секция protected: — поля и переменные попадают в protected_fields / protected_variables.
        let source = r#"
            cls Base {
                private:
                    a: int
                protected:
                    b: int
                    helper = 10
                public:
                    c: int
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, private_fields, protected_fields, protected_variables, public_fields, .. } = &stmts[0] {
            assert_eq!(name, "Base");
            assert_eq!(private_fields.len(), 1);
            assert_eq!(private_fields[0].name, "a");
            assert_eq!(protected_fields.len(), 1);
            assert_eq!(protected_fields[0].name, "b");
            assert_eq!(protected_variables.len(), 1);
            assert_eq!(protected_variables[0].name, "helper");
            assert_eq!(public_fields.len(), 1);
            assert_eq!(public_fields[0].name, "c");
        } else {
            panic!("Expected Class with protected section");
        }
    }

    #[test]
    fn test_parser_class_field_before_section_goes_to_private() {
        let source = r#"
            cls Parent {
                v: int
                public:
                new Parent(v) {
                    this.v = v
                }
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, private_fields, public_fields, constructors, .. } = &stmts[0] {
            assert_eq!(name, "Parent");
            assert_eq!(private_fields.len(), 1);
            assert_eq!(private_fields[0].name, "v");
            assert!(public_fields.is_empty());
            assert_eq!(constructors.len(), 1);
        } else {
            panic!("Expected Class with field before section in private_fields");
        }
    }

    #[test]
    fn test_parser_class_with_constructor() {
        let source = r#"
            cls Point {
                public:
                    x: int
                    y: int
                new Point(x, y) {
                    this.x = x
                    this.y = y
                }
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, constructors, .. } = &stmts[0] {
            assert_eq!(name, "Point");
            assert_eq!(constructors.len(), 1);
            assert_eq!(constructors[0].params.len(), 2);
            assert_eq!(constructors[0].params[0].name, "x");
            assert_eq!(constructors[0].params[1].name, "y");
            assert_eq!(constructors[0].body.len(), 2);
        } else {
            panic!("Expected Class with constructor");
        }
    }

    #[test]
    fn test_parser_class_with_method() {
        let source = r#"
            cls Counter {
                public:
                    n: int
                new Counter(n) { this.n = n }
                fn get() { return this.n }
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, methods, .. } = &stmts[0] {
            assert_eq!(name, "Counter");
            assert_eq!(methods.len(), 1);
            assert_eq!(methods[0].name, "get");
            assert_eq!(methods[0].params.len(), 0);
        } else {
            panic!("Expected Class with method");
        }
    }

    #[test]
    fn test_parser_class_constructor_name_must_match() {
        let source = r#"
            cls Foo {
                public:
                    x: int
                new Bar(x) { this.x = x }
            }
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let result = parser.parse();
        assert!(result.is_err(), "expected parse error for constructor name mismatch");
    }

    // ========== Парсер: наследование (superclass, delegate_args) ==========

    #[test]
    fn test_parser_class_with_superclass() {
        let source = r#"
            cls Child(Parent) {
                public:
                    x: int
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, superclass, public_fields, .. } = &stmts[0] {
            assert_eq!(name, "Child");
            assert_eq!(superclass.as_deref(), Some("Parent"));
            assert_eq!(public_fields.len(), 1);
            assert_eq!(public_fields[0].name, "x");
        } else {
            panic!("Expected Class with superclass, got {:?}", stmts[0]);
        }
    }

    #[test]
    fn test_parser_class_without_superclass_has_none() {
        let source = r#"
            cls Empty {
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, superclass, .. } = &stmts[0] {
            assert_eq!(name, "Empty");
            assert!(superclass.is_none());
        } else {
            panic!("Expected Class without superclass");
        }
    }

    #[test]
    fn test_parser_class_constructor_with_delegate_args() {
        let source = r#"
            cls Child(Parent) {
                public:
                    x: int
                new Child(path: str) : this(path) {
                }
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { name, constructors, .. } = &stmts[0] {
            assert_eq!(name, "Child");
            assert_eq!(constructors.len(), 1);
            let delegate_args = &constructors[0].delegate_args;
            assert!(delegate_args.is_some(), "expected delegate_args");
            let args = delegate_args.as_ref().unwrap();
            assert_eq!(args.len(), 1);
            if let data_code::parser::ast::Expr::Variable { name, .. } = &args[0] {
                assert_eq!(name, "path");
            } else {
                panic!("expected Variable in delegate_args, got {:?}", args[0]);
            }
        } else {
            panic!("Expected Class with delegating constructor");
        }
    }

    #[test]
    fn test_parser_class_constructor_without_delegate_has_none() {
        let source = r#"
            cls Point {
                public:
                    x: int
                    y: int
                new Point(x, y) {
                    this.x = x
                    this.y = y
                }
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Class { constructors, .. } = &stmts[0] {
            assert_eq!(constructors.len(), 1);
            assert!(constructors[0].delegate_args.is_none());
        } else {
            panic!("Expected Class with normal constructor");
        }
    }

    // ========== Интеграция: создание экземпляра и методы ==========

    #[test]
    fn test_class_simple_constructor_and_method() {
        let source = r#"
            cls Point {
                public:
                    x: int
                    y: int
                new Point(x, y) {
                    this.x = x
                    this.y = y
                }
                fn getX() { return this.x }
                fn getY() { return this.y }
            }
            let p = Point(10, 20)
            p.getX()
        "#;
        assert_number(source, 10.0);
    }

    #[test]
    fn test_class_private_variable_access_error() {
        // Приватная переменная уровня класса (вне public:) недоступна снаружи — ошибка доступа.
        let source = r#"
            cls Config {
                z = 10
                public:
                    x: int
                new Config(x) { this.x = x }
            }
            Config.z
        "#;
        assert_runtime_error(source, "cannot be accessed");
    }

    #[test]
    fn test_class_parent_private_variable_access_error() {
        // Попытка получить private переменную уровня класса у класса-родителя — ошибка доступа.
        let source = r#"
            cls Parent {
                secret = 42
                public:
                    v: int
                new Parent(v) { this.v = v }
            }
            cls Child(Parent) {
                fn getV() { return this.v }
                new Child(v: int) : this(v) { }
            }
            Parent.secret
        "#;
        assert_runtime_error(source, "cannot be accessed");
    }

    #[test]
    fn test_class_public_variable_on_class_object() {
        let source = r#"
            cls C {
                public:
                    tag = 99
            }
            C.tag
        "#;
        assert_number(source, 99.0);
    }

    #[test]
    fn test_class_method_get_y() {
        let source = r#"
            cls Point {
                public:
                    x: int
                    y: int
                new Point(x, y) {
                    this.x = x
                    this.y = y
                }
                fn getY() { return this.y }
            }
            let p = Point(1, 42)
            p.getY()
        "#;
        assert_number(source, 42.0);
    }

    #[test]
    fn test_class_private_field_and_method() {
        let source = r#"
            cls Counter {
                private:
                    n: int = 0
                public:
                    dummy: int
                new Counter() {
                    this.dummy = 0
                }
                fn inc() { this.n = this.n + 1 }
                fn get() { return this.n }
            }
            let c = Counter()
            c.inc()
            c.inc()
            c.get()
        "#;
        assert_number(source, 2.0);
    }

    #[test]
    fn test_class_deposit_and_balance() {
        let source = r#"
            cls Account {
                private:
                    balance: float = 0.0
                public:
                    name: str
                new Account(name) {
                    this.name = name
                }
                fn deposit(amount: float) {
                    this.balance = this.balance + amount
                }
                fn getBalance() { return this.balance }
            }
            let a = Account("test")
            a.deposit(100)
            a.deposit(50)
            a.getBalance()
        "#;
        assert_number(source, 150.0);
    }

    #[test]
    fn test_class_multiple_constructors_by_arity() {
        let source = r#"
            cls Data {
                private:
                    id: int
                    balance: float = 0.0
                public:
                    name: str
                    value: int
                new Data(id, name, value) {
                    this.id = id
                    this.name = name
                    this.value = value
                }
                new Data(id, balance, name, value) {
                    this.id = id
                    this.balance = balance
                    this.name = name
                    this.value = value
                }
                fn getBalance() { return this.balance }
            }
            let d1 = Data(1, "A", 10)
            let d2 = Data(2, 100.0, "B", 20)
            d1.getBalance()
        "#;
        assert_number(source, 0.0);
    }

    #[test]
    fn test_class_second_constructor_balance() {
        let source = r#"
            cls Data {
                private:
                    id: int
                    balance: float = 0.0
                public:
                    name: str
                    value: int
                new Data(id, name, value) {
                    this.id = id
                    this.name = name
                    this.value = value
                }
                new Data(id, balance, name, value) {
                    this.id = id
                    this.balance = balance
                    this.name = name
                    this.value = value
                }
                fn getBalance() { return this.balance }
            }
            let d = Data(1, 99.5, "B", 20)
            d.getBalance()
        "#;
        assert_number(source, 99.5);
    }

    #[test]
    fn test_class_method_no_args() {
        let source = r#"
            cls C {
                public:
                    x: int
                new C(x) { this.x = x }
                fn val() { return this.x }
            }
            let o = C(7)
            o.val()
        "#;
        assert_number(source, 7.0);
    }

    #[test]
    fn test_class_method_with_arg() {
        let source = r#"
            cls Calc {
                public:
                    base: int
                new Calc(base) { this.base = base }
                fn add(n) { return this.base + n }
            }
            let c = Calc(10)
            c.add(5)
        "#;
        assert_number(source, 15.0);
    }

    #[test]
    fn test_class_last_expression_is_result() {
        let source = r#"
            cls Box {
                public:
                    v: int
                new Box(v) { this.v = v }
                fn get() { return this.v }
            }
            let b = Box(33)
            b.get()
        "#;
        assert_number(source, 33.0);
    }

    #[test]
    fn test_class_program_returns_null_if_no_expression() {
        let source = r#"
            cls T {
                public:
                    x: int
                new T(x) { this.x = x }
            }
            let x = T(1)
        "#;
        assert_null(source);
    }

    #[test]
    fn test_class_two_instances_independent() {
        let source = r#"
            cls Counter {
                private:
                    n: int = 0
                public:
                    d: int
                new Counter() { this.d = 0 }
                fn inc() { this.n = this.n + 1 }
                fn get() { return this.n }
            }
            let a = Counter()
            let b = Counter()
            a.inc()
            a.inc()
            b.inc()
            a.get()
        "#;
        assert_number(source, 2.0);
    }

    #[test]
    fn test_class_public_field_access_after_construct() {
        let source = r#"
            cls Point {
                public:
                    x: int
                    y: int
                new Point(x, y) {
                    this.x = x
                    this.y = y
                }
            }
            let p = Point(3, 4)
            p.x + p.y
        "#;
        assert_number(source, 7.0);
    }

    /// Без классов: функция Base, вызов Base(10) — проверка, что глобалы и вызов работают.
    #[test]
    fn test_fn_base_call_no_class() {
        let source = r#"
            fn Base(x) {
                return { "value": x }
            }
            let b = Base(10)
            b.value
        "#;
        assert_number(source, 10.0);
    }

    // ========== Интеграция: наследование (implicit / delegating constructor) ==========

    #[test]
    fn test_class_implicit_constructor_inheritance() {
        // Родитель — функция, возвращающая объект. Наследник без конструкторов:
        // генерируется неявный конструктор с одним аргументом, вызов родителя, методы.
        let source = r#"
            fn Base(x) {
                return { "value": x }
            }
            
            cls Child(Base) {
                fn getValue() { return this.value }
            }
            let c = Child(10)
            c.getValue()
        "#;
        assert_number(source, 10.0);
    }

    #[test]
    fn test_class_explicit_delegating_constructor() {
        // Родитель — функция. Наследник с явным делегирующим конструктором : this(x).
        let source = r#"
            fn Base(x) {
                return { "v": x }
            }
            cls Child(Base) {
                fn getV() { return this.v }
                new Child(x) : this(x) {
                }
            }
            let c = Child(5)
            c.getV()
        "#;
        assert_number(source, 5.0);
    }

    // ========== Наследование от класса ==========

    #[test]
    fn test_parser_inheritance_from_class() {
        // Наследование от другого класса: Parent определён выше, Child(Parent).
        let source = r#"
            cls Parent {
                public:
                    v: int
                new Parent(v) { this.v = v }
            }
            cls Child(Parent) {
                fn getV() { return this.v }
                new Child(v: int) : this(v) { }
            }
        "#;
        let stmts = parse(source);
        assert_eq!(stmts.len(), 2);
        if let (Stmt::Class { name: pname, superclass: psuper, .. }, Stmt::Class { name: cname, superclass: csuper, constructors, .. }) = (&stmts[0], &stmts[1]) {
            assert_eq!(pname, "Parent");
            assert!(psuper.is_none());
            assert_eq!(cname, "Child");
            assert_eq!(csuper.as_deref(), Some("Parent"));
            assert_eq!(constructors.len(), 1);
            assert!(constructors[0].delegate_args.is_some());
        } else {
            panic!("Expected two classes: Parent, then Child(Parent)");
        }
    }

    #[test]
    fn test_class_inheritance_from_class_delegating_constructor() {
        // Наследование от класса: Parent с полем v, Child(Parent) с делегированием : this(v).
        // Родитель — класс, вызов суперкласса = вызов конструктора Parent::new_1(v).
        let source = r#"
            cls Parent {
                public:
                    v: int
                new Parent(v) { this.v = v }
            }
            cls Child(Parent) {
                fn getV() { return this.v }
                new Child(v: int) : this(v) { }
            }
            let c = Child(5)
            c.getV()
        "#;
        assert_number(source, 5.0);
    }

    #[test]
    fn test_class_inheritance_from_class_implicit_constructor() {
        // Наследование от класса без явных конструкторов у наследника — неявный конструктор с одним аргументом.
        let source = r#"
            cls Parent {
                public:
                    v: int
                new Parent(v) { this.v = v }
            }
            cls Child(Parent) {
                fn getV() { return this.v }
            }
            let c = Child(10)
            c.getV()
        "#;
        assert_number(source, 10.0);
    }

    // ========== Инкапсуляция (private / public) ==========

    #[test]
    fn test_encapsulation_public_field_accessible_from_outside() {
        let source = r#"
            cls Point {
                public:
                    x: int
                    y: int
                new Point(x, y) { this.x = x; this.y = y }
            }
            let p = Point(1, 2)
            p.x + p.y
        "#;
        assert_number(source, 3.0);
    }

    #[test]
    fn test_encapsulation_private_field_accessible_only_via_method() {
        // Приватное поле не должно быть доступно снаружи; доступ только через метод.
        let source = r#"
            cls Wallet {
                private:
                    balance: float = 0.0
                public:
                    name: str
                new Wallet(name) { this.name = name }
                fn add(amount) { this.balance = this.balance + amount }
                fn getBalance() { return this.balance }
            }
            let w = Wallet("mine")
            w.add(100)
            w.add(50)
            w.getBalance()
        "#;
        assert_number(source, 150.0);
    }

    #[test]
    fn test_encapsulation_private_field_direct_access_error() {
        // Прямой доступ к приватному полю экземпляра снаружи — ProtectError (not accessible).
        let source = r#"
            cls Wallet {
                private:
                    balance: float = 0.0
                public:
                    name: str
                new Wallet(name) { this.name = name }
                fn getBalance() { return this.balance }
            }
            let w = Wallet("mine")
            w.balance
        "#;
        assert_runtime_error(source, "cannot be accessed");
    }

    #[test]
    fn test_encapsulation_private_and_public_mixed() {
        // Класс с приватным счётчиком и публичным именем; снаружи видно только имя и методы.
        let source = r#"
            cls Counter {
                private:
                    n: int = 0
                public:
                    label: str
                new Counter(label) { this.label = label }
                fn inc() { this.n = this.n + 1 }
                fn get() { return this.n }
            }
            let c = Counter("A")
            c.inc()
            c.inc()
            c.label
        "#;
        // Проверяем, что публичное поле label доступно (результат последнего выражения — строка "A").
        let v = run_ok(source);
        assert!(matches!(v, Value::String(ref s) if s == "A"), "expected String(\"A\"), got {:?}", v);
    }

    #[test]
    fn test_encapsulation_runtime_error_on_invalid_access() {
        // Вызов не-функции даёт ожидаемую runtime-ошибку (проверка assert_runtime_error).
        let source = r#"
            let x = 1
            x()
        "#;
        assert_runtime_error(source, "Can only call functions");
    }

    #[test]
    fn test_encapsulation_two_instances_private_isolated() {
        // Два экземпляра — приватные поля изолированы.
        let source = r#"
            cls Box {
                private:
                    secret: int = 0
                public:
                    id: int
                new Box(id) { this.id = id }
                fn setSecret(v) { this.secret = v }
                fn getSecret() { return this.secret }
            }
            let a = Box(1)
            let b = Box(2)
            a.setSecret(10)
            b.setSecret(20)
            a.getSecret() + b.getSecret()
        "#;
        assert_number(source, 30.0);
    }

    // ========== Protected: парсер и рантайм ==========

    #[test]
    fn test_protected_field_accessible_inside_class() {
        // Protected-поле доступно внутри класса (в методе того же класса).
        let source = r#"
            cls Base {
                protected:
                    value: int
                public:
                new Base(v) { this.value = v }
                fn getValue() { return this.value }
            }
            let b = Base(42)
            b.getValue()
        "#;
        assert_number(source, 42.0);
    }

    #[test]
    fn test_protected_field_accessible_in_subclass() {
        // Protected-поле доступно в наследнике (метод наследника читает поле родителя).
        let source = r#"
            cls Base {
                protected:
                    value: int
                public:
                new Base(v) { this.value = v }
            }
            cls Derived(Base) {
                public:
                new Derived(v) : this(v) { }
                fn getBaseValue() { return this.value }
            }
            let d = Derived(100)
            d.getBaseValue()
        "#;
        assert_number(source, 100.0);
    }

    #[test]
    fn test_protected_field_not_accessible_from_outside() {
        // Доступ к protected-полю снаружи (глобальный код) — ProtectError (not accessible).
        let source = r#"
            cls Base {
                protected:
                    value: int
                public:
                new Base(v) { this.value = v }
            }
            let b = Base(42)
            b.value
        "#;
        assert_runtime_error(source, "cannot be accessed");
    }

    #[test]
    fn test_protected_field_not_accessible_from_unrelated_class() {
        // Метод другого класса не может читать protected-поле экземпляра.
        let source = r#"
            cls Base {
                protected:
                    value: int
                public:
                new Base(v) { this.value = v }
            }
            cls Other {
                public:
                new Other() { }
                fn readProtected(obj) { return obj.value }
            }
            let b = Base(99)
            let o = Other()
            o.readProtected(b)
        "#;
        assert_runtime_error(source, "cannot be accessed");
    }

    #[test]
    fn test_subclass_cannot_access_parent_private_field() {
        // Подкласс не может обращаться к приватному полю родителя — ProtectError.
        let source = r#"
            cls Parent {
                v: int
                protected:
                    p: int
                public:
                new Parent(v, p) { this.v = v; this.p = p }
            }
            cls Child(Parent) {
                public:
                new Child(v) {
                    super(v, 202)
                    this.v = v
                    this.p = 202
                }
            }
            Child(10)
        "#;
        assert_runtime_error(source, "private in 'Parent'");
    }
}
