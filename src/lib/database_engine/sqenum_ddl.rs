//! Pure SQL builders for PostgreSQL enum types (no live connection required).

/// Snake_case type name from a PascalCase class name (e.g. `UserRole` -> `user_role`).
pub fn enum_type_name_from_class_name(class_name: &str) -> String {
    let mut out = String::new();
    for (i, c) in class_name.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_lowercase().next().unwrap_or(c));
    }
    out
}

/// `CREATE TYPE name AS ENUM ('a', 'b', ...);` — values SQL-escaped as string literals.
pub fn create_type_enum_sql(type_name: &str, values: &[String]) -> String {
    let list = values
        .iter()
        .map(|s| format!("'{}'", s.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ");
    format!("CREATE TYPE {} AS ENUM ({})", type_name, list)
}

/// Column type fragment for a PostgreSQL enum (unquoted identifier; quote if needed by caller).
pub fn column_sql_pg_enum_type(type_name: &str) -> String {
    type_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_role_type_name() {
        assert_eq!(enum_type_name_from_class_name("UserRole"), "user_role");
    }

    #[test]
    fn create_type_golden() {
        let sql = create_type_enum_sql(
            "user_role",
            &["admin".into(), "user".into(), "moder".into()],
        );
        assert_eq!(
            sql,
            "CREATE TYPE user_role AS ENUM ('admin', 'user', 'moder')"
        );
    }

    #[test]
    fn create_type_escapes_quotes() {
        let sql = create_type_enum_sql("t", &["a'b".into()]);
        assert_eq!(sql, "CREATE TYPE t AS ENUM ('a''b')");
    }
}
