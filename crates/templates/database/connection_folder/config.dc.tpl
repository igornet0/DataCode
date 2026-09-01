database = {
    type = "{{ db_type }}"
    driver = "{{ driver }}"

    host = "{{ host }}"
    port = {{ port }}
    name = "{{ name }}"

    user = "{{ user }}"
    password = {{ password_expr }}

    pool = {
        enabled = {{ pool_enabled }}
        size = {{ pool_size }}
        max_overflow = {{ pool_max_overflow }}
    }

    async = {{ async }}
}
