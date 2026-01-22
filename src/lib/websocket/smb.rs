use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::fs;
use std::io::Write;
use serde::{Deserialize, Serialize};
use regex::Regex;

/// Параметры подключения к SMB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmbConnection {
    pub ip: String,
    pub login: String,
    pub password: String,
    pub domain: String,
    pub share_name: String,
}

impl SmbConnection {
    pub fn new(ip: String, login: String, password: String, domain: String, share_name: String) -> Self {
        Self {
            ip,
            login,
            password,
            domain,
            share_name,
        }
    }

    /// Получить UNC путь для подключения
    #[allow(dead_code)]
    pub fn get_unc_path(&self) -> String {
        format!("\\\\{}\\{}", self.ip, self.share_name)
    }

    /// Получить путь для lib:// протокола
    #[allow(dead_code)]
    pub fn get_lib_path(&self) -> String {
        format!("lib://{}", self.share_name)
    }
}

/// Менеджер SMB подключений для websocket сессий
pub struct SmbManager {
    /// Хранилище подключений по имени шары
    connections: HashMap<String, SmbConnection>,
    /// Временная директория для монтирования SMB
    mount_base: PathBuf,
}

impl SmbManager {
    pub fn new() -> Self {
        let mount_base = std::env::temp_dir().join("datacode_smb");
        // Создаем директорию если её нет
        let _ = fs::create_dir_all(&mount_base);
        
        Self {
            connections: HashMap::new(),
            mount_base,
        }
    }

    /// Подключиться к SMB шаре
    pub fn connect(&mut self, connection: SmbConnection) -> Result<String, String> {
        let share_name = connection.share_name.clone();
        
        // Проверяем, не подключены ли уже
        if self.connections.contains_key(&share_name) {
            return Err(format!("SMB share '{}' уже подключена", share_name));
        }

        // Пытаемся подключиться используя системные команды
        let mount_path = self.mount_base.join(&share_name);
        let _ = fs::create_dir_all(&mount_path);

        #[cfg(target_os = "windows")]
        {
            // Windows: используем net use
            let unc_path = connection.get_unc_path();
            let output = Command::new("net")
                .args(&["use", &unc_path, &format!("/user:{}\\{}", connection.domain, connection.login), &connection.password])
                .output()
                .map_err(|e| format!("Ошибка выполнения net use: {}", e))?;

            if !output.status.success() {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Ошибка подключения к SMB: {}", error_msg));
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Linux/Mac: используем smbclient или mount.cifs
            // Сначала проверяем доступность smbclient
            let smbclient_check = Command::new("which")
                .arg("smbclient")
                .output();

            if smbclient_check.is_ok() && smbclient_check.unwrap().status.success() {
                // Пытаемся проверить подключение, но не строго
                // Если проверка не удалась, все равно сохраняем подключение
                // и будем проверять при реальных операциях с файлами
                let user_string = if connection.domain.is_empty() {
                    connection.login.clone()
                } else {
                    format!("{}\\{}", connection.domain, connection.login)
                };
                
                let mut args = vec![
                    "-L".to_string(),
                    connection.ip.clone(),
                    "-U".to_string(),
                    user_string.clone(),
                ];
                
                // Добавляем -W только если домен не пустой
                if !connection.domain.is_empty() {
                    args.push("-W".to_string());
                    args.push(connection.domain.clone());
                }
                
                // Пробуем передать пароль через stdin
                let mut cmd = Command::new("smbclient");
                cmd.args(&args);
                let child = cmd.stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn();
                
                if let Ok(mut child_process) = child {
                    if let Some(mut stdin) = child_process.stdin.take() {
                        let _ = writeln!(stdin, "{}", connection.password);
                    }
                    
                    let output = child_process.wait_with_output();
                    if let Ok(out) = output {
                        if !out.status.success() {
                            // Проверка не удалась, но это не критично
                            // Сохраняем подключение и проверим при реальных операциях
                            let error_msg = String::from_utf8_lossy(&out.stderr);
                            // Не возвращаем ошибку, просто логируем (если нужно)
                            eprintln!("Предупреждение: проверка подключения не удалась: {}", error_msg);
                        }
                    }
                }
                // В любом случае продолжаем и сохраняем подключение
            } else {
                // Пытаемся использовать mount.cifs если доступен
                let mount_check = Command::new("which")
                    .arg("mount.cifs")
                    .output();

                if mount_check.is_ok() && mount_check.unwrap().status.success() {
                    // Создаем credentials файл
                    let creds_file = self.mount_base.join(format!("{}.creds", share_name));
                    let mut creds = fs::File::create(&creds_file)
                        .map_err(|e| format!("Не удалось создать файл credentials: {}", e))?;
                    
                    writeln!(creds, "username={}", connection.login)
                        .map_err(|e| format!("Ошибка записи в credentials файл: {}", e))?;
                    writeln!(creds, "password={}", connection.password)
                        .map_err(|e| format!("Ошибка записи в credentials файл: {}", e))?;
                    writeln!(creds, "domain={}", connection.domain)
                        .map_err(|e| format!("Ошибка записи в credentials файл: {}", e))?;

                    // Монтируем шару
                    let unc_path = format!("//{}/{}", connection.ip, connection.share_name);
                    // Получаем uid и gid текущего пользователя
                    let uid = std::env::var("UID")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(1000);
                    let gid = std::env::var("GID")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(1000);

                    let output = Command::new("sudo")
                        .args(&[
                            "mount",
                            "-t", "cifs",
                            &unc_path,
                            &mount_path.to_string_lossy(),
                            "-o", &format!("credentials={},uid={},gid={}", 
                                creds_file.to_string_lossy(),
                                uid,
                                gid),
                        ])
                        .output()
                        .map_err(|e| format!("Ошибка выполнения mount: {}", e))?;

                    if !output.status.success() {
                        let error_msg = String::from_utf8_lossy(&output.stderr);
                        return Err(format!("Ошибка монтирования SMB: {}", error_msg));
                    }
                } else {
                    // Если нет ни smbclient, ни mount.cifs, просто сохраняем подключение
                    // и будем использовать smbclient для операций с файлами
                }
            }
        }

        // Сохраняем подключение
        self.connections.insert(share_name.clone(), connection);

        Ok(format!("Успешно подключено к SMB шаре '{}'", share_name))
    }

    /// Получить подключение по имени шары
    #[allow(dead_code)]
    pub fn get_connection(&self, share_name: &str) -> Option<&SmbConnection> {
        self.connections.get(share_name)
    }

    /// Проверить, подключена ли шара
    #[allow(dead_code)]
    pub fn is_connected(&self, share_name: &str) -> bool {
        self.connections.contains_key(share_name)
    }

    /// Конвертирует glob паттерн в regex (для SMB)
    fn glob_to_regex_smb(glob: &str) -> String {
        // Если паттерн содержит |, разделяем на альтернативы
        if glob.contains('|') {
            let alternatives: Vec<&str> = glob.split('|').collect();
            let regex_alternatives: Vec<String> = alternatives
                .iter()
                .map(|alt| Self::glob_to_regex_smb_single(alt.trim()))
                .collect();
            return format!("^({})$", regex_alternatives.join("|"));
        }
        
        // Для одиночного паттерна добавляем якоря
        format!("^{}$", Self::glob_to_regex_smb_single(glob))
    }

    /// Конвертирует один glob паттерн (без |) в regex
    /// Не добавляет якоря ^ и $ - они добавляются в вызывающей функции
    fn glob_to_regex_smb_single(glob: &str) -> String {
        let mut regex = String::new();
        let mut chars = glob.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                '*' => {
                    if chars.peek() == Some(&'*') {
                        chars.next();
                        regex.push_str(".*");
                    } else {
                        regex.push_str(".*");
                    }
                }
                '?' => {
                    regex.push('.');
                }
                '.' | '(' | ')' | '[' | ']' | '{' | '}' | '+' | '^' | '$' | '\\' => {
                    // Экранируем специальные символы regex (кроме |, который обрабатывается отдельно)
                    regex.push('\\');
                    regex.push(ch);
                }
                _ => {
                    regex.push(ch);
                }
            }
        }
        
        regex
    }

    /// Рекурсивно обходит SMB директорию и собирает все файлы
    #[cfg(not(target_os = "windows"))]
    fn list_files_recursive_smb(
        &self,
        connection: &SmbConnection,
        base_path: &str,
        regex: &Option<Regex>,
    ) -> Result<Vec<String>, String> {
        let mut all_files = Vec::new();
        let mut dirs_to_process = vec![base_path.to_string()];
        
        while let Some(current_dir) = dirs_to_process.pop() {
            // Формируем команду для smbclient
            let smb_command = if current_dir.is_empty() || current_dir == "/" {
                "ls".to_string()
            } else {
                format!("cd {}; ls", current_dir)
            };

            // Формируем строку пользователя для smbclient
            let user_string = if connection.domain.is_empty() {
                connection.login.clone()
            } else {
                format!("{}\\{}", connection.domain, connection.login)
            };

            let mut cmd = Command::new("smbclient");
            let mut args = vec![
                format!("//{}/{}", connection.ip, connection.share_name),
                "-U".to_string(),
                user_string.clone(),
                "-c".to_string(),
                smb_command,
            ];
            
            // Добавляем -W только если домен не пустой
            if !connection.domain.is_empty() {
                args.push("-W".to_string());
                args.push(connection.domain.clone());
            }
            
            cmd.args(&args);

            // Используем stdin для передачи пароля
            let mut child = cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| format!("Ошибка запуска smbclient: {}", e))?;

            if let Some(mut stdin) = child.stdin.take() {
                let _ = writeln!(stdin, "{}", connection.password);
            }

            let output = child.wait_with_output()
                .map_err(|e| format!("Ошибка выполнения smbclient: {}", e))?;

            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let original_line = line;
                    let trimmed = original_line.trim();
                    
                    // Пропускаем служебные строки
                    if trimmed.is_empty() 
                        || trimmed.starts_with('[') 
                        || trimmed.contains("blocks")
                        || trimmed.starts_with("Password")
                        || trimmed.starts_with("Can't")
                        || trimmed.starts_with("Try \"")
                        || trimmed == "." 
                        || trimmed == ".."
                        || original_line.starts_with("\t") {
                        continue;
                    }
                    
                    // Парсим вывод smbclient
                    if original_line.starts_with(' ') {
                        let mut type_char_pos = None;
                        let chars: Vec<char> = original_line.chars().collect();
                        
                        for i in 2..chars.len().saturating_sub(2) {
                            if (chars[i] == 'A' || chars[i] == 'D') && chars[i-1] == ' ' {
                                if i > 10 && (chars.get(i+1) == Some(&' ') || chars.get(i+1) == Some(&'H')) {
                                    let spaces_before = (0..i).rev().take_while(|&j| chars[j] == ' ').count();
                                    if spaces_before >= 5 {
                                        type_char_pos = Some(i);
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if let Some(char_pos) = type_char_pos {
                            let byte_pos = original_line
                                .char_indices()
                                .nth(char_pos)
                                .map(|(idx, _)| idx)
                                .unwrap_or(original_line.len());
                            
                            let file_name = original_line
                                .char_indices()
                                .take_while(|(idx, _)| *idx < byte_pos)
                                .map(|(_, ch)| ch)
                                .collect::<String>()
                                .trim()
                                .to_string();
                            
                            let file_type: String = chars.iter().skip(char_pos).take_while(|c| **c != ' ').collect();
                            let is_directory = file_type == "D";
                            
                            if !file_name.is_empty() 
                                && file_name != "." 
                                && file_name != ".."
                                && !file_name.starts_with("._")
                                && file_name != ".DS_Store" {
                                
                                // Формируем полный путь
                                let full_path = if current_dir.is_empty() || current_dir == "/" {
                                    file_name.clone()
                                } else {
                                    format!("{}/{}", current_dir, file_name)
                                };
                                
                                if is_directory {
                                    // Добавляем директорию в очередь для обработки
                                    dirs_to_process.push(full_path);
                                } else {
                                    // Применяем regex фильтрацию, если задана
                                    if let Some(ref re) = regex {
                                        if !re.is_match(&file_name) {
                                            continue;
                                        }
                                    }
                                    all_files.push(full_path);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(all_files)
    }

    /// Получить список файлов из SMB шары
    /// 
    /// # Arguments
    /// * `share_name` - Имя подключенной SMB шары
    /// * `path` - Путь к директории на шаре
    /// * `regex` - Опциональный regex паттерн для фильтрации имен файлов
    /// * `recursive` - Если true, рекурсивно обходит поддиректории
    pub fn list_files(&self, share_name: &str, path: &str, regex: Option<&str>, recursive: bool) -> Result<Vec<String>, String> {
        let connection = self.connections.get(share_name)
            .ok_or_else(|| format!("SMB share '{}' не подключена", share_name))?;

        // Компилируем regex, если он задан
        let regex = if let Some(pattern) = regex {
            // Конвертируем glob паттерн в regex, если нужно
            let regex_pattern_str = if pattern.contains('*') || pattern.contains('?') {
                Self::glob_to_regex_smb(pattern)
            } else {
                pattern.to_string()
            };
            
            match Regex::new(&regex_pattern_str) {
                Ok(re) => Some(re),
                Err(e) => return Err(format!("Invalid regex pattern '{}': {}", pattern, e)),
            }
        } else {
            None
        };

        let mut files = Vec::new();

        #[cfg(target_os = "windows")]
        {
            // Windows: используем dir команду
            let full_path = if path.is_empty() || path == "/" {
                connection.get_unc_path()
            } else {
                format!("{}\\{}", connection.get_unc_path(), path.replace("/", "\\"))
            };

            // Для рекурсивного обхода используем /S флаг
            let dir_args = if recursive {
                vec!["/C", "dir", "/B", "/S", &full_path]
            } else {
                vec!["/C", "dir", "/B", &full_path]
            };

            let output = Command::new("cmd")
                .args(&dir_args)
                .output()
                .map_err(|e| format!("Ошибка выполнения dir: {}", e))?;

            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let base_unc_path = connection.get_unc_path();
                
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if recursive {
                            // Для рекурсивного обхода dir /B /S возвращает полные UNC пути
                            // Преобразуем их в относительные пути от base_path
                            let relative_path = if trimmed.starts_with(&base_unc_path) {
                                // Убираем базовый UNC путь и начальный слеш
                                let path_part = &trimmed[base_unc_path.len()..];
                                path_part.trim_start_matches(|c| c == '\\' || c == '/').to_string()
                            } else {
                                // Если путь не начинается с базового, используем как есть
                                trimmed.to_string()
                            };
                            
                            // Извлекаем имя файла для regex фильтрации
                            let file_name = PathBuf::from(&relative_path)
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or(&relative_path)
                                .to_string();
                            
                            // Применяем regex фильтрацию, если задана
                            if let Some(ref re) = regex {
                                if !re.is_match(&file_name) {
                                    continue;
                                }
                            }
                            
                            // Возвращаем относительный путь
                            files.push(relative_path);
                        } else {
                            // Для нерекурсивного обхода просто имя файла
                            let file_name = trimmed.to_string();
                            
                            // Применяем regex фильтрацию, если задана
                            if let Some(ref re) = regex {
                                if !re.is_match(&file_name) {
                                    continue;
                                }
                            }
                            
                            files.push(file_name);
                        }
                    }
                }
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Ошибка получения списка файлов: {}", error_msg));
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Linux/Mac: используем smbclient
            // Проверяем наличие smbclient
            let smbclient_check = Command::new("which")
                .arg("smbclient")
                .output();
            
            if smbclient_check.is_err() || !smbclient_check.unwrap().status.success() {
                return Err(format!("smbclient не найден. Установите его через: brew install samba"));
            }
            
            // Для рекурсивного обхода используем рекурсивную функцию
            if recursive {
                return self.list_files_recursive_smb(connection, path, &regex);
            }
            
            // Формируем команду для smbclient
            // Если путь не пустой, сначала переходим в директорию, затем выполняем ls
            let smb_command = if path.is_empty() || path == "/" {
                "ls".to_string()
            } else {
                // Используем cd для перехода в директорию, затем ls
                format!("cd {}; ls", path)
            };

            // Формируем строку пользователя для smbclient
            let user_string = if connection.domain.is_empty() {
                connection.login.clone()
            } else {
                format!("{}\\{}", connection.domain, connection.login)
            };

            let mut cmd = Command::new("smbclient");
            let mut args = vec![
                format!("//{}/{}", connection.ip, connection.share_name),
                "-U".to_string(),
                user_string.clone(),
                "-c".to_string(),
                smb_command,
            ];
            
            // Добавляем -W только если домен не пустой
            if !connection.domain.is_empty() {
                args.push("-W".to_string());
                args.push(connection.domain.clone());
            }
            
            cmd.args(&args);

            // Используем stdin для передачи пароля (переменная окружения не работает)
            let mut child = cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| format!("Ошибка запуска smbclient: {}", e))?;

            if let Some(mut stdin) = child.stdin.take() {
                let _ = writeln!(stdin, "{}", connection.password);
            }

            let output = child.wait_with_output()
                .map_err(|e| format!("Ошибка выполнения smbclient: {}", e))?;

            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    // Проверяем исходную строку до trim для определения формата
                    let original_line = line;
                    let trimmed = original_line.trim();
                    
                    // Пропускаем служебные строки
                    if trimmed.is_empty() 
                        || trimmed.starts_with('[') 
                        || trimmed.contains("blocks")
                        || trimmed.starts_with("Password")
                        || trimmed.starts_with("Can't")
                        || trimmed.starts_with("Try \"")
                        || trimmed == "." 
                        || trimmed == ".."
                        || original_line.starts_with("\t") {
                        continue;
                    }
                    
                    // Парсим вывод smbclient
                    // Формат: "  filename with spaces                    A    size  date time" (файл)
                    // или:    "  dirname                    D        0  date time" (директория)
                    // Имя файла начинается после начальных пробелов и заканчивается перед типом (A/D)
                    // Проверяем исходную строку, так как после trim пробелы исчезают
                    if original_line.starts_with(' ') {
                        // Ищем позицию типа файла (A, D, AH и т.д.)
                        // Тип находится после имени файла, которое заканчивается множеством пробелов
                        // Ищем паттерн: пробелы, затем A или D, затем пробел или H
                        let mut type_char_pos = None;
                        let chars: Vec<char> = original_line.chars().collect();
                        
                        for i in 2..chars.len().saturating_sub(2) {
                            // Ищем последовательность: пробел(ы) + A/D + пробел/H
                            if (chars[i] == 'A' || chars[i] == 'D') && chars[i-1] == ' ' {
                                // Проверяем, что перед этим было достаточно пробелов (имя файла закончилось)
                                // И что после типа идет пробел или H
                                if i > 10 && (chars.get(i+1) == Some(&' ') || chars.get(i+1) == Some(&'H')) {
                                    // Проверяем, что перед типом было много пробелов (минимум 5)
                                    let spaces_before = (0..i).rev().take_while(|&j| chars[j] == ' ').count();
                                    if spaces_before >= 5 {
                                        type_char_pos = Some(i);
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if let Some(char_pos) = type_char_pos {
                            // Находим позицию в байтах для char_pos-го символа
                            let byte_pos = original_line
                                .char_indices()
                                .nth(char_pos)
                                .map(|(idx, _)| idx)
                                .unwrap_or(original_line.len());
                            
                            // Извлекаем имя файла до позиции типа, используя безопасную индексацию
                            let file_name = original_line
                                .char_indices()
                                .take_while(|(idx, _)| *idx < byte_pos)
                                .map(|(_, ch)| ch)
                                .collect::<String>()
                                .trim()
                                .to_string();
                            
                            // Определяем тип файла
                            let file_type: String = chars.iter().skip(char_pos).take_while(|c| **c != ' ').collect();
                            let _is_directory = file_type == "D";
                            
                            // Возвращаем и файлы, и директории (кроме служебных)
                            // Пропускаем скрытые файлы (начинающиеся с точки) и служебные
                            if !file_name.is_empty() 
                                && file_name != "." 
                                && file_name != ".."
                                && !file_name.starts_with("._")
                                && file_name != ".DS_Store" {
                                // Применяем regex фильтрацию, если задана
                                if let Some(ref re) = regex {
                                    if !re.is_match(&file_name) {
                                        continue;
                                    }
                                }
                                files.push(file_name);
                            }
                        }
                    }
                }
            } else {
                // Попробуем альтернативный способ с передачей пароля через stdin
                let user_string = if connection.domain.is_empty() {
                    connection.login.clone()
                } else {
                    format!("{}\\{}", connection.domain, connection.login)
                };
                
                // Формируем команду для smbclient
                let smb_command = if path.is_empty() || path == "/" {
                    "ls".to_string()
                } else {
                    format!("cd {}; ls", path)
                };
                
                let mut cmd = Command::new("smbclient");
                let mut args = vec![
                    format!("//{}/{}", connection.ip, connection.share_name),
                    "-U".to_string(),
                    user_string.clone(),
                    "-c".to_string(),
                    smb_command,
                ];
                
                // Добавляем -W только если домен не пустой
                if !connection.domain.is_empty() {
                    args.push("-W".to_string());
                    args.push(connection.domain.clone());
                }
                
                cmd.args(&args);

                let mut child = cmd.stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("Ошибка запуска smbclient: {}", e))?;

                if let Some(mut stdin) = child.stdin.take() {
                    let _ = writeln!(stdin, "{}", connection.password);
                }

                let output = child.wait_with_output()
                    .map_err(|e| format!("Ошибка выполнения smbclient: {}", e))?;

                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        // Проверяем исходную строку до trim для определения формата
                        let original_line = line;
                        let trimmed = original_line.trim();
                        
                        // Пропускаем служебные строки
                        if trimmed.is_empty() 
                            || trimmed.starts_with('[') 
                            || trimmed.contains("blocks")
                            || trimmed.starts_with("Password")
                            || trimmed.starts_with("Can't")
                            || trimmed.starts_with("Try \"")
                            || trimmed == "." 
                            || trimmed == ".."
                            || original_line.starts_with("\t") {
                            continue;
                        }
                        
                        // Парсим вывод smbclient
                        // Формат: "  filename with spaces                    A    size  date time" (файл)
                        // или:    "  dirname                    D        0  date time" (директория)
                        // Имя файла начинается после начальных пробелов и заканчивается перед типом (A/D)
                        // Проверяем исходную строку, так как после trim пробелы исчезают
                        if original_line.starts_with(' ') {
                            // Ищем позицию типа файла (A, D, AH и т.д.)
                            // Тип находится после имени файла, которое заканчивается множеством пробелов
                            // Ищем паттерн: пробелы, затем A или D, затем пробел или H
                            let mut type_char_pos = None;
                            let chars: Vec<char> = original_line.chars().collect();
                            
                            for i in 2..chars.len().saturating_sub(2) {
                                // Ищем последовательность: пробел(ы) + A/D + пробел/H
                                if (chars[i] == 'A' || chars[i] == 'D') && chars[i-1] == ' ' {
                                    // Проверяем, что перед этим было достаточно пробелов (имя файла закончилось)
                                    // И что после типа идет пробел или H
                                    if i > 10 && (chars.get(i+1) == Some(&' ') || chars.get(i+1) == Some(&'H')) {
                                        // Проверяем, что перед типом было много пробелов (минимум 5)
                                        let spaces_before = (0..i).rev().take_while(|&j| chars[j] == ' ').count();
                                        if spaces_before >= 5 {
                                            type_char_pos = Some(i);
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            if let Some(char_pos) = type_char_pos {
                                // Находим позицию в байтах для char_pos-го символа
                                let byte_pos = original_line
                                    .char_indices()
                                    .nth(char_pos)
                                    .map(|(idx, _)| idx)
                                    .unwrap_or(original_line.len());
                                
                                // Извлекаем имя файла до позиции типа, используя безопасную индексацию
                                let file_name = original_line
                                    .char_indices()
                                    .take_while(|(idx, _)| *idx < byte_pos)
                                    .map(|(_, ch)| ch)
                                    .collect::<String>()
                                    .trim()
                                    .to_string();
                                
                                // Определяем тип файла
                                let file_type: String = chars.iter().skip(char_pos).take_while(|c| **c != ' ').collect();
                                let _is_directory = file_type == "D";
                                
                                // Возвращаем и файлы, и директории (кроме служебных)
                                // Пропускаем скрытые файлы (начинающиеся с точки) и служебные
                                if !file_name.is_empty() 
                                    && file_name != "." 
                                    && file_name != ".."
                                    && !file_name.starts_with("._")
                                    && file_name != ".DS_Store" {
                                    files.push(file_name);
                                }
                            }
                        }
                    }
                } else {
                    let error_msg = String::from_utf8_lossy(&output.stderr);
                    return Err(format!("Ошибка получения списка файлов: {}", error_msg));
                }
            }
        }

        Ok(files)
    }

    /// Прочитать файл из SMB шары
    pub fn read_file(&self, share_name: &str, file_path: &str) -> Result<Vec<u8>, String> {
        let connection = self.connections.get(share_name)
            .ok_or_else(|| format!("SMB share '{}' не подключена", share_name))?;

        #[cfg(target_os = "windows")]
        {
            // Windows: читаем через UNC путь
            let full_path = format!("{}\\{}", connection.get_unc_path(), file_path.replace("/", "\\"));
            fs::read(&full_path)
                .map_err(|e| format!("Ошибка чтения файла: {}", e))
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Linux/Mac: используем smbclient get
            // Проверяем наличие smbclient
            let smbclient_check = Command::new("which")
                .arg("smbclient")
                .output();
            
            if smbclient_check.is_err() || !smbclient_check.unwrap().status.success() {
                return Err(format!("smbclient не найден. Установите его через: brew install samba"));
            }
            
            let user_string = if connection.domain.is_empty() {
                connection.login.clone()
            } else {
                format!("{}\\{}", connection.domain, connection.login)
            };
            
            let mut cmd = Command::new("smbclient");
            let mut args = vec![
                format!("//{}/{}", connection.ip, connection.share_name),
                "-U".to_string(),
                user_string.clone(),
                "-c".to_string(),
                format!("get \"{}\" -", file_path),
            ];
            
            // Добавляем -W только если домен не пустой
            if !connection.domain.is_empty() {
                args.push("-W".to_string());
                args.push(connection.domain.clone());
            }
            
            cmd.args(&args);

            let mut child = cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| format!("Ошибка запуска smbclient: {}", e))?;

            if let Some(mut stdin) = child.stdin.take() {
                let _ = writeln!(stdin, "{}", connection.password);
            }

            let output = child.wait_with_output()
                .map_err(|e| format!("Ошибка выполнения smbclient: {}", e))?;

            if output.status.success() {
                Ok(output.stdout)
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                Err(format!("Ошибка чтения файла: {}", error_msg))
            }
        }
    }

    /// Отключиться от SMB шары
    pub fn disconnect(&mut self, share_name: &str) -> Result<String, String> {
        if !self.connections.contains_key(share_name) {
            return Err(format!("SMB share '{}' не подключена", share_name));
        }

        #[cfg(target_os = "windows")]
        {
            let connection = self.connections.get(share_name).unwrap();
            let unc_path = connection.get_unc_path();
            let _ = Command::new("net")
                .args(&["use", &unc_path, "/delete", "/yes"])
                .output();
        }

        #[cfg(not(target_os = "windows"))]
        {
            let mount_path = self.mount_base.join(share_name);
            if mount_path.exists() {
                let _ = Command::new("sudo")
                    .args(&["umount", &mount_path.to_string_lossy()])
                    .output();
            }
        }

        self.connections.remove(share_name);
        Ok(format!("Отключено от SMB шары '{}'", share_name))
    }

    /// Получить все подключенные шары
    pub fn list_connections(&self) -> Vec<String> {
        self.connections.keys().cloned().collect()
    }
}

impl Default for SmbManager {
    fn default() -> Self {
        Self::new()
    }
}

