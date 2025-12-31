/**
 * DataCode Syntax Highlighter
 * Provides syntax highlighting for DataCode language
 */

class DataCodeSyntaxHighlighter {
    constructor() {
        // DataCode keywords
        this.keywords = [
            'global', 'let', 'fn',
            'if', 'else',
            'for', 'in', 'while',
            'return', 'try', 'catch', 'finally', 'throw',
            'break', 'continue',
            'import', 'from', 'as', 'cache',
            'true', 'false', 'null'
        ];

        // Built-in functions (from DataCode registry)
        this.builtinFunctions = [
            // System
            'print', 'now', 'getcwd', 'isinstance', 'isset',
            // Type conversion
            'int', 'float', 'bool', 'str', 'date', 'money', 'typeof', 'array',
            // File operations
            'read_file', 'write_file', 'list_files', 'path',
            // Math
            'abs', 'sqrt', 'pow', 'min', 'max', 'round', 'div',
            'sum', 'avg',
            // Array operations
            'len', 'length', 'push', 'pop', 'sort', 'unique', 'average',
            // String operations
            'split', 'join', 'trim', 'upper', 'lower', 'contains',
            // Table operations
            'table', 'show_table', 'table_info', 'table_select',
            'table_headers', 'table_head', 'merge_tables', 'relate',
            // Filter operations
            'table_where', 'table_filter', 'table_distinct', 'table_sample',
            'table_between', 'table_in',
            // Iteration
            'enum'
        ];

        // Operators
        this.operators = ['+', '-', '*', '/', '//', '**', '%', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not', ':'];

        // Function definitions with parameters and descriptions
        this.functionDefinitions = this.initializeFunctionDefinitions();
    }

    initializeFunctionDefinitions() {
        // Based on definitions.ts from datacode-syntax extension
        return {
            // System functions
            'now': {
                signature: 'now()',
                description: 'Возвращает текущую дату и время в формате RFC3339',
                parameters: [],
                returnType: 'string',
                category: 'system'
            },
            'print': {
                signature: 'print(...values)',
                description: 'Выводит значения в консоль, разделенные пробелами',
                parameters: [
                    { name: 'values', type: 'any', description: 'Значения для вывода (переменное количество)' }
                ],
                returnType: 'null',
                category: 'system'
            },
            'getcwd': {
                signature: 'getcwd()',
                description: 'Возвращает текущую рабочую директорию как путь',
                parameters: [],
                returnType: 'path',
                category: 'system'
            },
            'isinstance': {
                signature: 'isinstance(value, type)',
                description: 'Проверяет, является ли значение определенного типа',
                parameters: [
                    { name: 'value', type: 'any', description: 'Значение для проверки' },
                    { name: 'type', type: 'string', description: 'Имя типа для проверки' }
                ],
                returnType: 'boolean',
                category: 'system'
            },
            'isset': {
                signature: 'isset(variable)',
                description: 'Проверяет, определена ли переменная и не равна null',
                parameters: [
                    { name: 'variable', type: 'any', description: 'Переменная для проверки' }
                ],
                returnType: 'boolean',
                category: 'system'
            },
            'str': {
                signature: 'str(value)',
                description: 'Преобразует значение в строковое представление',
                parameters: [
                    { name: 'value', type: 'any', description: 'Значение для преобразования в строку' }
                ],
                returnType: 'string',
                category: 'system'
            },
            // File operations
            'path': {
                signature: 'path(string_path)',
                description: 'Создает объект пути из строки',
                parameters: [
                    { name: 'string_path', type: 'string', description: 'Строковое представление пути' }
                ],
                returnType: 'path',
                category: 'file'
            },
            'list_files': {
                signature: 'list_files(directory_path)',
                description: 'Списывает файлы в директории или соответствует glob-шаблону',
                parameters: [
                    { name: 'directory_path', type: 'path', description: 'Путь к директории или glob-шаблон' }
                ],
                returnType: 'array',
                category: 'file'
            },
            'read_file': {
                signature: 'read_file(file_path, header_row?, sheet_name?)',
                description: 'Читает файл (.csv, .xlsx) и возвращает его содержимое или создает таблицу. Для XLSX можно указать имя листа. Для CSV/XLSX можно указать номер строки заголовка (начиная с 0)',
                parameters: [
                    { name: 'file_path', type: 'path', description: 'Путь к файлу для чтения' },
                    { name: 'header_row', type: 'number', description: 'Номер строки заголовка, начиная с 0 (по умолчанию: 0)', optional: true },
                    { name: 'sheet_name', type: 'string', description: 'Имя листа для XLSX файлов (по умолчанию: первый лист)', optional: true }
                ],
                returnType: 'string | table',
                category: 'file'
            },
            // Math functions
            'abs': {
                signature: 'abs(number)',
                description: 'Возвращает абсолютное значение числа',
                parameters: [
                    { name: 'number', type: 'number', description: 'Число для получения абсолютного значения' }
                ],
                returnType: 'number',
                category: 'math'
            },
            'sqrt': {
                signature: 'sqrt(number)',
                description: 'Возвращает квадратный корень числа',
                parameters: [
                    { name: 'number', type: 'number', description: 'Число для извлечения квадратного корня (должно быть неотрицательным)' }
                ],
                returnType: 'number',
                category: 'math'
            },
            'pow': {
                signature: 'pow(base, exponent)',
                description: 'Возвращает основание, возведенное в степень показателя',
                parameters: [
                    { name: 'base', type: 'number', description: 'Основание' },
                    { name: 'exponent', type: 'number', description: 'Показатель степени' }
                ],
                returnType: 'number',
                category: 'math'
            },
            'min': {
                signature: 'min(array)',
                description: 'Возвращает минимальное значение из массива чисел',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив чисел' }
                ],
                returnType: 'number',
                category: 'math'
            },
            'max': {
                signature: 'max(array)',
                description: 'Возвращает максимальное значение из массива чисел',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив чисел' }
                ],
                returnType: 'number',
                category: 'math'
            },
            'round': {
                signature: 'round(number, decimals)',
                description: 'Округляет число до указанного количества знаков после запятой',
                parameters: [
                    { name: 'number', type: 'number', description: 'Число для округления' },
                    { name: 'decimals', type: 'integer', description: 'Количество знаков после запятой', optional: true }
                ],
                returnType: 'number',
                category: 'math'
            },
            'div': {
                signature: 'div(dividend, divisor)',
                description: 'Выполняет деление с проверкой на ноль',
                parameters: [
                    { name: 'dividend', type: 'number', description: 'Делимое' },
                    { name: 'divisor', type: 'number', description: 'Делитель (не может быть нулем)' }
                ],
                returnType: 'number',
                category: 'math'
            },
            // Array operations
            'length': {
                signature: 'length(array)',
                description: 'Возвращает длину массива',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив для получения длины' }
                ],
                returnType: 'integer',
                category: 'array'
            },
            'len': {
                signature: 'len(array)',
                description: 'Алиас для length() - возвращает длину массива',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив для получения длины' }
                ],
                returnType: 'integer',
                category: 'array'
            },
            'push': {
                signature: 'push(array, element)',
                description: 'Добавляет элемент в конец массива',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив для добавления элемента' },
                    { name: 'element', type: 'any', description: 'Элемент для добавления' }
                ],
                returnType: 'null',
                category: 'array'
            },
            'pop': {
                signature: 'pop(array)',
                description: 'Удаляет и возвращает последний элемент массива',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив для удаления элемента' }
                ],
                returnType: 'any',
                category: 'array'
            },
            'sort': {
                signature: 'sort(array)',
                description: 'Сортирует массив по возрастанию',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив для сортировки' }
                ],
                returnType: 'null',
                category: 'array'
            },
            'unique': {
                signature: 'unique(array)',
                description: 'Возвращает новый массив с удаленными дубликатами',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив для удаления дубликатов' }
                ],
                returnType: 'array',
                category: 'array'
            },
            'sum': {
                signature: 'sum(array)',
                description: 'Возвращает сумму всех чисел в массиве',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив чисел для суммирования' }
                ],
                returnType: 'number',
                category: 'array'
            },
            'average': {
                signature: 'average(array)',
                description: 'Возвращает среднее значение всех чисел в массиве',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив чисел для вычисления среднего' }
                ],
                returnType: 'number',
                category: 'array'
            },
            // String functions
            'split': {
                signature: 'split(string, delimiter)',
                description: 'Разделяет строку на массив с использованием разделителя',
                parameters: [
                    { name: 'string', type: 'string', description: 'Строка для разделения' },
                    { name: 'delimiter', type: 'string', description: 'Разделитель для разделения' }
                ],
                returnType: 'array',
                category: 'string'
            },
            'join': {
                signature: 'join(array, delimiter)',
                description: 'Объединяет массив строк в одну строку',
                parameters: [
                    { name: 'array', type: 'array', description: 'Массив строк для объединения' },
                    { name: 'delimiter', type: 'string', description: 'Разделитель для объединения' }
                ],
                returnType: 'string',
                category: 'string'
            },
            'trim': {
                signature: 'trim(string)',
                description: 'Удаляет пробелы в начале и конце строки',
                parameters: [
                    { name: 'string', type: 'string', description: 'Строка для обрезки' }
                ],
                returnType: 'string',
                category: 'string'
            },
            'upper': {
                signature: 'upper(string)',
                description: 'Преобразует строку в верхний регистр',
                parameters: [
                    { name: 'string', type: 'string', description: 'Строка для преобразования' }
                ],
                returnType: 'string',
                category: 'string'
            },
            'lower': {
                signature: 'lower(string)',
                description: 'Преобразует строку в нижний регистр',
                parameters: [
                    { name: 'string', type: 'string', description: 'Строка для преобразования' }
                ],
                returnType: 'string',
                category: 'string'
            },
            'contains': {
                signature: 'contains(string, substring)',
                description: 'Проверяет, содержит ли строка подстроку',
                parameters: [
                    { name: 'string', type: 'string', description: 'Строка для поиска' },
                    { name: 'substring', type: 'string', description: 'Подстрока для поиска' }
                ],
                returnType: 'boolean',
                category: 'string'
            },
            // Table operations
            'table': {
                signature: 'table(data, headers)',
                description: 'Создает таблицу из массива данных и заголовков',
                parameters: [
                    { name: 'data', type: 'array', description: 'Массив массивов, представляющих строки таблицы' },
                    { name: 'headers', type: 'array', description: 'Массив имен заголовков столбцов' }
                ],
                returnType: 'table',
                category: 'table'
            },
            'show_table': {
                signature: 'show_table(table)',
                description: 'Отображает таблицу в форматированном ASCII-выводе',
                parameters: [
                    { name: 'table', type: 'table', description: 'Таблица для отображения' }
                ],
                returnType: 'null',
                category: 'table'
            },
            'table_info': {
                signature: 'table_info(table)',
                description: 'Возвращает информацию о таблице (строки, столбцы, типы)',
                parameters: [
                    { name: 'table', type: 'table', description: 'Таблица для получения информации' }
                ],
                returnType: 'object',
                category: 'table'
            },
            'table_select': {
                signature: 'table_select(table, columns)',
                description: 'Выбирает определенные столбцы из таблицы',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'columns', type: 'array', description: 'Массив имен столбцов для выбора' }
                ],
                returnType: 'table',
                category: 'table'
            },
            'table_headers': {
                signature: 'table_headers(table)',
                description: 'Возвращает заголовки столбцов таблицы',
                parameters: [
                    { name: 'table', type: 'table', description: 'Таблица для получения заголовков' }
                ],
                returnType: 'array',
                category: 'table'
            },
            'table_head': {
                signature: 'table_head(table, count)',
                description: 'Возвращает первые N строк таблицы',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'count', type: 'integer', description: 'Количество строк для возврата', optional: true }
                ],
                returnType: 'table',
                category: 'table'
            },
            'table_where': {
                signature: 'table_where(table, column, operator, value)',
                description: 'Фильтрует строки таблицы на основе условия',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'column', type: 'string', description: 'Имя столбца для фильтрации' },
                    { name: 'operator', type: 'string', description: 'Оператор сравнения (>, <, ==, !=, >=, <=)' },
                    { name: 'value', type: 'any', description: 'Значение для сравнения' }
                ],
                returnType: 'table',
                category: 'table'
            },
            'table_filter': {
                signature: 'table_filter(table, condition)',
                description: 'Фильтрует строки таблицы с использованием пользовательского условия в виде строки',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'condition', type: 'string', description: 'Условие фильтрации в виде строкового выражения' }
                ],
                returnType: 'table',
                category: 'filter'
            },
            'table_distinct': {
                signature: 'table_distinct(table, column)',
                description: 'Возвращает уникальные значения из столбца таблицы',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'column', type: 'string', description: 'Имя столбца для получения уникальных значений' }
                ],
                returnType: 'array',
                category: 'filter'
            },
            'table_sample': {
                signature: 'table_sample(table, count)',
                description: 'Возвращает случайную выборку строк из таблицы',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'count', type: 'integer', description: 'Количество строк для выборки' }
                ],
                returnType: 'table',
                category: 'filter'
            },
            'table_between': {
                signature: 'table_between(table, column, min_value, max_value)',
                description: 'Фильтрует строки таблицы, где значение столбца находится между min и max',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'column', type: 'string', description: 'Имя столбца для фильтрации' },
                    { name: 'min_value', type: 'any', description: 'Минимальное значение (включительно)' },
                    { name: 'max_value', type: 'any', description: 'Максимальное значение (включительно)' }
                ],
                returnType: 'table',
                category: 'filter'
            },
            'table_in': {
                signature: 'table_in(table, column, values)',
                description: 'Фильтрует строки таблицы, где значение столбца находится в заданном списке',
                parameters: [
                    { name: 'table', type: 'table', description: 'Исходная таблица' },
                    { name: 'column', type: 'string', description: 'Имя столбца для фильтрации' },
                    { name: 'values', type: 'array', description: 'Массив значений для сопоставления' }
                ],
                returnType: 'table',
                category: 'filter'
            },
            // Iteration
            'enum': {
                signature: 'enum(iterable)',
                description: 'Возвращает перечисленные пары (индекс, значение) для итерации',
                parameters: [
                    { name: 'iterable', type: 'array | table', description: 'Массив или таблица для перечисления' }
                ],
                returnType: 'array',
                category: 'iteration'
            },
            // Type conversion functions
            'int': {
                signature: 'int(value)',
                description: 'Преобразует значение в целое число',
                parameters: [
                    { name: 'value', type: 'any', description: 'Значение для преобразования в целое число' }
                ],
                returnType: 'integer',
                category: 'type'
            },
            'float': {
                signature: 'float(value)',
                description: 'Преобразует значение в число с плавающей точкой',
                parameters: [
                    { name: 'value', type: 'any', description: 'Значение для преобразования в число с плавающей точкой' }
                ],
                returnType: 'float',
                category: 'type'
            },
            'bool': {
                signature: 'bool(value)',
                description: 'Преобразует значение в логическое значение',
                parameters: [
                    { name: 'value', type: 'any', description: 'Значение для преобразования в логическое значение' }
                ],
                returnType: 'boolean',
                category: 'type'
            },
            'date': {
                signature: 'date(value)',
                description: 'Преобразует значение в дату',
                parameters: [
                    { name: 'value', type: 'string', description: 'Строковое представление даты (RFC3339 или YYYY-MM-DD)' }
                ],
                returnType: 'date',
                category: 'type'
            },
            'money': {
                signature: 'money(amount, format)',
                description: 'Преобразует число в денежное значение с указанным форматом',
                parameters: [
                    { name: 'amount', type: 'number', description: 'Числовое значение суммы' },
                    { name: 'format', type: 'string', description: 'Формат отображения (например, "$0.00", "0,0 $")' }
                ],
                returnType: 'money',
                category: 'type'
            },
            'typeof': {
                signature: 'typeof(value)',
                description: 'Возвращает строку с типом значения',
                parameters: [
                    { name: 'value', type: 'any', description: 'Значение для определения типа' }
                ],
                returnType: 'string',
                category: 'type'
            },
            'array': {
                signature: 'array(...values)',
                description: 'Создает массив из переданных значений',
                parameters: [
                    { name: 'values', type: 'any', description: 'Значения для создания массива (переменное количество)' }
                ],
                returnType: 'array',
                category: 'type'
            }
        };
    }

    /**
     * Get function definition by name
     */
    getFunctionDefinition(name) {
        return this.functionDefinitions[name.toLowerCase()] || null;
    }

    /**
     * Highlight DataCode code
     * @param {string} code - The code to highlight
     * @returns {string} - HTML with syntax highlighting
     */
    highlight(code) {
        if (!code) return '';

        // Split into lines first for better handling
        const lines = code.split('\n');
        return lines.map(line => this.highlightLine(line)).join('\n');
    }

    highlightLine(line) {
        if (!line) return '';

        let html = '';
        let i = 0;
        const len = line.length;

        while (i < len) {
            // Check for comments (comments go to end of line)
            if (line[i] === '#') {
                const comment = line.substring(i);
                html += `<span class="comment">${this.escapeHtml(comment)}</span>`;
                break; // Rest of line is comment
            }

            // Check for strings (single quotes)
            if (line[i] === "'" && (i === 0 || line[i - 1] !== '\\')) {
                let stringEnd = i + 1;
                while (stringEnd < len) {
                    if (line[stringEnd] === "'" && line[stringEnd - 1] !== '\\') {
                        stringEnd++;
                        break;
                    }
                    stringEnd++;
                }
                const str = line.substring(i, stringEnd);
                html += `<span class="string">${this.escapeHtml(str)}</span>`;
                i = stringEnd;
                continue;
            }

            // Check for strings (double quotes)
            if (line[i] === '"' && (i === 0 || line[i - 1] !== '\\')) {
                let stringEnd = i + 1;
                while (stringEnd < len) {
                    if (line[stringEnd] === '"' && line[stringEnd - 1] !== '\\') {
                        stringEnd++;
                        break;
                    }
                    stringEnd++;
                }
                const str = line.substring(i, stringEnd);
                html += `<span class="string">${this.escapeHtml(str)}</span>`;
                i = stringEnd;
                continue;
            }

            // Check for numbers
            if (this.isDigit(line[i]) || (line[i] === '-' && i + 1 < len && this.isDigit(line[i + 1]))) {
                let numEnd = i + 1;
                let hasDot = line[i] === '.';
                
                while (numEnd < len) {
                    if (this.isDigit(line[numEnd])) {
                        numEnd++;
                    } else if (line[numEnd] === '.' && !hasDot) {
                        hasDot = true;
                        numEnd++;
                    } else if (this.isIdentifierChar(line[numEnd])) {
                        numEnd++;
                    } else {
                        break;
                    }
                }
                
                // Make sure we're not part of an identifier
                if (i > 0 && this.isIdentifierChar(line[i - 1])) {
                    html += this.escapeHtml(line[i]);
                    i++;
                    continue;
                }
                
                const num = line.substring(i, numEnd);
                html += `<span class="number">${this.escapeHtml(num)}</span>`;
                i = numEnd;
                continue;
            }

            // Check for identifiers (keywords, functions, variables)
            if (this.isIdentifierStart(line[i])) {
                let identEnd = i + 1;
                while (identEnd < len && this.isIdentifierChar(line[identEnd])) {
                    identEnd++;
                }
                
                const ident = line.substring(i, identEnd);
                const lowerIdent = ident.toLowerCase();
                
                // Check if it's a keyword
                if (this.keywords.includes(lowerIdent)) {
                    html += `<span class="keyword">${this.escapeHtml(ident)}</span>`;
                }
                // Check if it's a built-in function
                else if (this.builtinFunctions.includes(lowerIdent)) {
                    html += `<span class="builtin" data-function="${this.escapeHtml(lowerIdent)}">${this.escapeHtml(ident)}</span>`;
                }
                // Check if it's a function call (has parentheses after)
                else if (identEnd < len && line[identEnd] === '(') {
                    // Check if it's a built-in function being called
                    if (this.builtinFunctions.includes(lowerIdent)) {
                        html += `<span class="builtin" data-function="${this.escapeHtml(lowerIdent)}">${this.escapeHtml(ident)}</span>`;
                    } else {
                        html += `<span class="function">${this.escapeHtml(ident)}</span>`;
                    }
                }
                // Otherwise it's a variable
                else {
                    html += `<span class="variable">${this.escapeHtml(ident)}</span>`;
                }
                
                i = identEnd;
                continue;
            }

            // Check for braces (object delimiters)
            if (line[i] === '{' || line[i] === '}') {
                html += `<span class="operator">${this.escapeHtml(line[i])}</span>`;
                i++;
                continue;
            }

            // Check for operators
            let operatorMatched = false;
            for (const op of this.operators.sort((a, b) => b.length - a.length)) {
                if (line.substring(i, i + op.length) === op) {
                    // Make sure it's not part of an identifier
                    if (i > 0 && this.isIdentifierChar(line[i - 1])) {
                        break;
                    }
                    // Make sure it's not part of a longer identifier
                    if (i + op.length < len && this.isIdentifierChar(line[i + op.length])) {
                        break;
                    }
                    html += `<span class="operator">${this.escapeHtml(op)}</span>`;
                    i += op.length;
                    operatorMatched = true;
                    break;
                }
            }
            
            if (operatorMatched) continue;

            // Default: escape and add
            html += this.escapeHtml(line[i]);
            i++;
        }

        return html;
    }

    /**
     * Check if character is a digit
     */
    isDigit(ch) {
        return ch >= '0' && ch <= '9';
    }

    /**
     * Check if character can start an identifier
     */
    isIdentifierStart(ch) {
        return (ch >= 'a' && ch <= 'z') || 
               (ch >= 'A' && ch <= 'Z') || 
               ch === '_';
    }

    /**
     * Check if character can be part of an identifier
     */
    isIdentifierChar(ch) {
        return this.isIdentifierStart(ch) || this.isDigit(ch);
    }

    /**
     * Escape HTML special characters
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Get tokens for autocomplete
     */
    getTokens(code) {
        const tokens = new Set();
        const regex = /\b([a-zA-Z_][a-zA-Z0-9_]*)\b/g;
        let match;
        
        while ((match = regex.exec(code)) !== null) {
            const token = match[1].toLowerCase();
            if (!this.keywords.includes(token) && !this.builtinFunctions.includes(token)) {
                tokens.add(match[1]);
            }
        }
        
        return Array.from(tokens);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataCodeSyntaxHighlighter;
}
