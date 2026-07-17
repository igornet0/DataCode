// Helper function to scroll to target with proper offset
function scrollToTarget(target) {
    const headerOffset = 120; // Account for fixed header
    const elementPosition = target.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

    window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
    });
}

// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    // Sidebar toggle functionality
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebarWrapper = document.querySelector('.sidebar-wrapper');
    
    if (sidebarToggle && sidebarWrapper) {
        // Check if mobile device
        const isMobile = window.matchMedia('(max-width: 768px)').matches;
        
        sidebarToggle.addEventListener('click', function() {
            sidebarWrapper.classList.toggle('collapsed');
            document.body.classList.toggle('sidebar-collapsed', sidebarWrapper.classList.contains('collapsed'));
            // Save state to localStorage
            localStorage.setItem('sidebarCollapsed', sidebarWrapper.classList.contains('collapsed'));
        });
        
        // Restore sidebar state from localStorage or default to collapsed on mobile
        const savedState = localStorage.getItem('sidebarCollapsed');
        if (savedState === 'true' || (isMobile && savedState === null)) {
            sidebarWrapper.classList.add('collapsed');
            document.body.classList.add('sidebar-collapsed');
        }
    }

    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            const targetContent = document.getElementById(targetTab);
            if (targetContent) {
                targetContent.classList.add('active');
                
                // Clear active TOC items when switching tabs
                const tocLinks = document.querySelectorAll('#toc-nav a');
                tocLinks.forEach(link => link.classList.remove('active'));
                
                // Scroll to top of the content area when switching tabs
                const headerOffset = 120;
                const contentTop = targetContent.getBoundingClientRect().top + window.pageYOffset - headerOffset;
                window.scrollTo({
                    top: Math.max(0, contentTop),
                    behavior: 'smooth'
                });
                
                // Rebuild TOC for new tab
                buildTOC(targetContent);
                
                // Setup scroll spy for the new tab
                setTimeout(() => {
                    setupTOCScrollSpy(targetContent);
                    // Update TOC active state after scroll (activate first heading)
                    const firstHeading = targetContent.querySelector('h2, h3');
                    if (firstHeading && tocLinks.length > 0) {
                        const firstLink = document.querySelector(`#toc-nav a[href="#${firstHeading.id}"]`);
                        if (firstLink) {
                            firstLink.classList.add('active');
                        }
                    }
                }, 200);
                
                // Setup function filtering if we're on functions tab
                if (targetTab === 'functions') {
                    // Wait a bit for DOM to update
                    setTimeout(() => {
                        setupFunctionFiltering();
                    }, 250);
                }
            }
        });
    });

    // Initialize TOC for first active tab
    const activeTab = document.querySelector('.tab-content.active');
    if (activeTab) {
        buildTOC(activeTab);
        setTimeout(() => {
            setupTOCScrollSpy(activeTab);
        }, 100);
    }

    // Load functions data (will setup filtering after loading)
    loadFunctions();
    
    // Also setup filtering if functions tab is initially active
    if (activeTab && activeTab.id === 'functions') {
        setTimeout(() => {
            setupFunctionFiltering();
        }, 200);
    }

    // Add scroll handler to update TOC on scroll
    let scrollTimeout;
    window.addEventListener('scroll', () => {
        // Throttle scroll events
        if (scrollTimeout) {
            clearTimeout(scrollTimeout);
        }
        scrollTimeout = setTimeout(() => {
            const activeTabContent = document.querySelector('.tab-content.active');
            if (activeTabContent) {
                updateTOCOnScroll(activeTabContent);
            }
        }, 50);
    }, { passive: true });

    // Smooth scroll for anchor links (excluding TOC links which have their own handler)
    document.querySelectorAll('a[href^="#"]:not(#toc-nav a)').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                // Find which tab contains this target
                const tabContent = target.closest('.tab-content');
                if (tabContent) {
                    const tabId = tabContent.id;
                    // Switch to the correct tab if not active
                    if (!tabContent.classList.contains('active')) {
                        const tabButton = document.querySelector(`[data-tab="${tabId}"]`);
                        if (tabButton) {
                            tabButton.click();
                            setTimeout(() => {
                                scrollToTarget(target);
                            }, 100);
                        } else {
                            scrollToTarget(target);
                        }
                    } else {
                        scrollToTarget(target);
                    }
                } else {
                    scrollToTarget(target);
                }
            }
        });
    });
});

// Build Table of Contents
function buildTOC(container) {
    const tocNav = document.getElementById('toc-nav');
    if (!tocNav) return;

    const headings = container.querySelectorAll('h2, h3');
    if (headings.length === 0) {
        tocNav.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem;">Нет заголовков</p>';
        return;
    }

    let tocHTML = '';
    headings.forEach((heading, index) => {
        const id = heading.id || `heading-${index}`;
        if (!heading.id) {
            heading.id = id;
        }

        const level = heading.tagName === 'H2' ? 2 : 3;
        const text = heading.textContent;
        const className = `toc-level-${level}`;

        tocHTML += `<li><a href="#${id}" class="${className}">${text}</a></li>`;
    });

    tocNav.innerHTML = tocHTML;

    // Add click handlers
    tocNav.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                // Find which tab contains this target
                const tabContent = target.closest('.tab-content');
                if (tabContent) {
                    const tabId = tabContent.id;
                    // Switch to the correct tab if not active
                    if (!tabContent.classList.contains('active')) {
                        const tabButton = document.querySelector(`[data-tab="${tabId}"]`);
                        if (tabButton) {
                            tabButton.click();
                            // Wait for tab switch animation and TOC rebuild before scrolling
                            setTimeout(() => {
                                // Find target again after tab switch (in case DOM changed)
                                const newTarget = document.getElementById(targetId);
                                if (newTarget) {
                                    scrollToTarget(newTarget);
                                    // Update active TOC item after rebuild
                                    const tocLinks = document.querySelectorAll('#toc-nav a');
                                    tocLinks.forEach(a => {
                                        if (a.getAttribute('href') === `#${targetId}`) {
                                            a.classList.add('active');
                                        } else {
                                            a.classList.remove('active');
                                        }
                                    });
                                }
                            }, 150);
                        } else {
                            scrollToTarget(target);
                            // Update active TOC item
                            tocNav.querySelectorAll('a').forEach(a => a.classList.remove('active'));
                            this.classList.add('active');
                        }
                    } else {
                        scrollToTarget(target);
                        // Update active TOC item
                        tocNav.querySelectorAll('a').forEach(a => a.classList.remove('active'));
                        this.classList.add('active');
                    }
                } else {
                    scrollToTarget(target);
                    // Update active TOC item
                    tocNav.querySelectorAll('a').forEach(a => a.classList.remove('active'));
                    this.classList.add('active');
                }
            }
        });
    });
}

// Store current observer to clean it up when switching tabs
let currentTOCObserver = null;

// Update TOC active item based on scroll position
function updateTOCOnScroll(container) {
    // Make sure we're working with the active tab content
    const activeTabContent = document.querySelector('.tab-content.active');
    if (!activeTabContent || container !== activeTabContent) {
        return;
    }

    const headings = container.querySelectorAll('h2, h3');
    const tocLinks = document.querySelectorAll('#toc-nav a');
    
    if (headings.length === 0 || tocLinks.length === 0) return;

    const headerOffset = 120;
    const scrollPosition = window.pageYOffset + headerOffset + 50;

    let currentActive = null;
    
    // Find the heading that is currently in view
    // Check from bottom to top to get the most recent heading
    for (let i = headings.length - 1; i >= 0; i--) {
        const heading = headings[i];
        const headingRect = heading.getBoundingClientRect();
        const headingTop = headingRect.top + window.pageYOffset;
        
        // Check if heading is visible and above the scroll position
        if (headingTop <= scrollPosition && headingRect.bottom > headerOffset) {
            currentActive = heading;
            break;
        }
    }

    // If no heading found and we're at the top, use the first one
    if (!currentActive && headings.length > 0) {
        if (window.pageYOffset < 200) {
            currentActive = headings[0];
        }
    }

    // Update active state in TOC
    if (currentActive) {
        const id = currentActive.id;
        tocLinks.forEach(link => {
            if (link.getAttribute('href') === `#${id}`) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
}

// Setup scroll spy for TOC
function setupTOCScrollSpy(container) {
    // Clean up previous observer if exists
    if (currentTOCObserver) {
        currentTOCObserver.disconnect();
        currentTOCObserver = null;
    }

    // Make sure we're working with the active tab content
    const activeTabContent = document.querySelector('.tab-content.active');
    if (!activeTabContent || container !== activeTabContent) {
        return;
    }

    const headings = container.querySelectorAll('h2, h3');
    const tocLinks = document.querySelectorAll('#toc-nav a');

    if (headings.length === 0 || tocLinks.length === 0) return;

    const observerOptions = {
        rootMargin: '-120px 0px -66%',
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1]
    };

    currentTOCObserver = new IntersectionObserver((entries) => {
        // Make sure we're still on the same active tab
        const currentActiveTab = document.querySelector('.tab-content.active');
        if (!currentActiveTab || currentActiveTab !== container) {
            return;
        }

        // Find the heading that is most visible and closest to the top
        let mostVisible = null;
        let maxVisibility = 0;
        let closestToTop = null;
        let minDistanceToTop = Infinity;

        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const visibility = entry.intersectionRatio;
                const distanceToTop = Math.abs(entry.boundingClientRect.top - 120);
                
                // Track most visible
                if (visibility > maxVisibility) {
                    maxVisibility = visibility;
                    mostVisible = entry.target;
                }
                
                // Track closest to top (within viewport)
                if (entry.boundingClientRect.top >= 100 && entry.boundingClientRect.top <= 200) {
                    if (distanceToTop < minDistanceToTop) {
                        minDistanceToTop = distanceToTop;
                        closestToTop = entry.target;
                    }
                }
            }
        });

        // Prefer closest to top, then most visible
        const activeHeading = closestToTop || mostVisible;

        // If we have an active heading, activate its TOC link
        if (activeHeading) {
            const id = activeHeading.id;
            tocLinks.forEach(link => {
                if (link.getAttribute('href') === `#${id}`) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        } else {
            // Fallback: find the first heading that is intersecting
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    tocLinks.forEach(link => {
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        } else {
                            link.classList.remove('active');
                        }
                    });
                }
            });
        }
    }, observerOptions);

    headings.forEach(heading => currentTOCObserver.observe(heading));
}

// Copy code functionality
function copyCode(button) {
    const codeBlock = button.closest('.code-block');
    const code = codeBlock.querySelector('pre code').textContent;

    navigator.clipboard.writeText(code).then(() => {
        const originalText = button.textContent;
        button.textContent = '✓ Скопировано!';
        button.classList.add('copied');

        setTimeout(() => {
            button.textContent = originalText;
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy code:', err);
        button.textContent = '❌ Ошибка';
        setTimeout(() => {
            button.textContent = '📋 Копировать';
        }, 2000);
    });
}

// Functions data
const functionsData = {
    system: [
        {
            name: 'print(...values)',
            category: 'system',
            description: 'Выводит значения в консоль, разделенные пробелами.',
            signature: 'print(value1, value2, ...)',
            example: "print('Hello', 'World', 42, true)"
        },
        {
            name: 'now()',
            category: 'system',
            description: 'Возвращает текущую дату и время в формате RFC3339.',
            signature: 'now()',
            example: "global current_time = now()\nprint('Current time:', current_time)"
        },
        {
            name: 'getcwd()',
            category: 'system',
            description: 'Возвращает текущую рабочую директорию как path объект.',
            signature: 'getcwd()',
            example: "global current_dir = getcwd()\nprint('Working directory:', current_dir)"
        },
        {
            name: 'isinstance(value, type)',
            category: 'system',
            description: 'Проверяет, является ли значение определенного типа.',
            signature: 'isinstance(value, type)',
            example: "if isinstance(age, integer) do\n    print('Age is an integer')\nendif"
        },
        {
            name: 'int(value)',
            category: 'system',
            description: 'Преобразует значение в целое число.',
            signature: 'int(value)',
            example: "global num = int('42')\nglobal whole = int(3.14)  # 3"
        },
        {
            name: 'float(value)',
            category: 'system',
            description: 'Преобразует значение в число с плавающей точкой.',
            signature: 'float(value)',
            example: "global num = float('3.14')\nglobal decimal = float(42)  # 42.0"
        },
        {
            name: 'bool(value)',
            category: 'system',
            description: 'Преобразует значение в булево значение.',
            signature: 'bool(value)',
            example: "global flag = bool(1)  # true\nglobal empty = bool('')  # false"
        },
        {
            name: 'date(value)',
            category: 'system',
            description: 'Преобразует строку в дату (проверяет валидность формата даты).',
            signature: 'date(value)',
            example: "global d = date('2024-12-31')\nglobal d2 = date('31.12.2024')"
        },
        {
            name: 'money(value, format?)',
            category: 'system',
            description: 'Преобразует значение в денежный формат. Опционально можно указать формат валюты.',
            signature: 'money(value, [format])',
            example: "global price = money(100.5)  # $100.50\nglobal euro = money(50, 'EUR')"
        },
        {
            name: 'typeof(value)',
            category: 'system',
            description: 'Возвращает строку с названием типа значения.',
            signature: 'typeof(value)',
            example: "global type1 = typeof(42)  # 'int'\nglobal type2 = typeof('hello')  # 'string'"
        },
        {
            name: 'isset(variable)',
            category: 'system',
            description: 'Проверяет, определена ли переменная и не равна ли null.',
            signature: 'isset(variable)',
            example: "if isset(my_var) do\n    print('Variable is set')\nendif"
        },
        {
            name: 'str(value)',
            category: 'system',
            description: 'Преобразует значение в строковое представление.',
            signature: 'str(value)',
            example: "global text = str(42)\nglobal array_str = str([1, 2, 3])"
        },
        {
            name: 'array(...values)',
            category: 'system',
            description: 'Создает массив из переданных аргументов.',
            signature: 'array(value1, value2, ...)',
            example: "global arr = array(1, 2, 3)  # [1, 2, 3]\nglobal mixed = array('a', 42, true)  # ['a', 42, true]"
        }
    ],
    file: [
        {
            name: 'path(string_path)',
            category: 'file',
            description: 'Создает path объект из строки.',
            signature: 'path(string_path)',
            example: "global file_path = path('/home/user/data.csv')\nglobal relative_path = path('data.csv')"
        },
        {
            name: 'list_files(directory_path)',
            category: 'file',
            description: 'Возвращает список файлов в директории или по glob паттерну.',
            signature: 'list_files(directory_path)',
            example: "global files = list_files(path('.'))\nglobal csv_files = list_files(path('*.csv'))"
        },
        {
            name: 'read(file_path)',
            category: 'file',
            description: 'Читает файл и возвращает содержимое или создает таблицу для CSV/Excel. Поддерживает опциональные параметры: read(path, sheet_name), read(path, header_row), read(path, header_row, sheet_name).',
            signature: 'read(path, [header_row], [sheet_name])',
            example: "global data = read(path('data.csv'))\nglobal text = read(path('readme.txt'))\nglobal excel = read(path('report.xlsx'), 'Sales')"
        },
        {
            name: 'analyze_csv(file_path)',
            category: 'file',
            description: 'Анализирует CSV файл и возвращает информацию о структуре.',
            signature: 'analyze_csv(file_path)',
            example: "global analysis = analyze_csv(path('data.csv'))\nprint('CSV structure:', analysis)"
        },
        {
            name: 'read_csv_safe(file_path)',
            category: 'file',
            description: 'Безопасно читает CSV файл с обработкой ошибок.',
            signature: 'read_csv_safe(file_path)',
            example: "global data = read_csv_safe(path('data.csv'))"
        },
        {
            name: 'path_name(path)',
            category: 'file',
            description: 'Возвращает имя файла или директории из пути.',
            signature: 'path_name(path)',
            example: "global name = path_name(path('data.csv'))  # 'data.csv'\nglobal file = path_name(path('/home/user/file.txt'))  # 'file.txt'"
        },
        {
            name: 'path_parent(path)',
            category: 'file',
            description: 'Возвращает родительский путь.',
            signature: 'path_parent(path)',
            example: "global parent = path_parent(path('folder/file.txt'))  # path('folder')\nglobal dir = path_parent(path('/home/user'))  # path('/home')"
        },
        {
            name: 'path_exists(path)',
            category: 'file',
            description: 'Проверяет существование файла или директории.',
            signature: 'path_exists(path)',
            example: "if path_exists(path('data.csv')) do\n    print('File exists')\nendif"
        },
        {
            name: 'path_is_file(path)',
            category: 'file',
            description: 'Проверяет, является ли путь файлом.',
            signature: 'path_is_file(path)',
            example: "if path_is_file(path('data.csv')) do\n    print('This is a file')\nendif"
        },
        {
            name: 'path_is_dir(path)',
            category: 'file',
            description: 'Проверяет, является ли путь директорией.',
            signature: 'path_is_dir(path)',
            example: "if path_is_dir(path('folder')) do\n    print('This is a directory')\nendif"
        },
        {
            name: 'path_extension(path)',
            category: 'file',
            description: 'Возвращает расширение файла (без точки).',
            signature: 'path_extension(path)',
            example: "global ext = path_extension(path('data.csv'))  # 'csv'\nglobal txt = path_extension(path('file.txt'))  # 'txt'"
        },
        {
            name: 'path_stem(path)',
            category: 'file',
            description: 'Возвращает имя файла без расширения.',
            signature: 'path_stem(path)',
            example: "global stem = path_stem(path('data.csv'))  # 'data'\nglobal name = path_stem(path('file.txt'))  # 'file'"
        },
        {
            name: 'path_len(path)',
            category: 'file',
            description: 'Возвращает длину строкового представления пути.',
            signature: 'path_len(path)',
            example: "global length = path_len(path('data.csv'))  # 8\nglobal len = path_len(path('folder/file'))  # 12"
        }
    ],
    math: [
        {
            name: 'abs(number)',
            category: 'math',
            description: 'Возвращает абсолютное значение числа.',
            signature: 'abs(number)',
            example: "global result = abs(-5)      # 5\nglobal result2 = abs(3.14)   # 3.14"
        },
        {
            name: 'sqrt(number)',
            category: 'math',
            description: 'Возвращает квадратный корень числа.',
            signature: 'sqrt(number)',
            example: "global result = sqrt(16)     # 4\nglobal result2 = sqrt(2.0)   # 1.414..."
        },
        {
            name: 'pow(base, exponent)',
            category: 'math',
            description: 'Возводит число в степень.',
            signature: 'pow(base, exponent)',
            example: "global result = pow(2, 3)    # 8\nglobal result2 = pow(10, 0.5) # 3.162..."
        },
        {
            name: 'min(array)',
            category: 'math',
            description: 'Возвращает минимальное значение из массива.',
            signature: 'min(array)',
            example: "global minimum = min([1, 5, 3, 9, 2])  # 1"
        },
        {
            name: 'max(array)',
            category: 'math',
            description: 'Возвращает максимальное значение из массива.',
            signature: 'max(array)',
            example: "global maximum = max([1, 5, 3, 9, 2])  # 9"
        },
        {
            name: 'round(number, decimals?)',
            category: 'math',
            description: 'Округляет число до указанного количества знаков после запятой.',
            signature: 'round(number, [decimals])',
            example: "global rounded = round(3.14159)     # 3\nglobal precise = round(3.14159, 2)  # 3.14"
        },
        {
            name: 'div(dividend, divisor)',
            category: 'math',
            description: 'Выполняет деление с проверкой на ноль.',
            signature: 'div(dividend, divisor)',
            example: "global result = div(10, 2)   # 5\nglobal safe = div(7, 3)      # 2.333..."
        }
    ],
    array: [
        {
            name: 'length(array) / len(array)',
            category: 'array',
            description: 'Возвращает длину массива или строки.',
            signature: 'length(array) / len(array)',
            example: "global size = length([1, 2, 3])  # 3\nglobal count = len(my_array)"
        },
        {
            name: 'push(array, element) / append(array, element)',
            category: 'array',
            description: 'Добавляет элемент в конец массива.',
            signature: 'push(array, element) / append(array, element)',
            example: "push(my_array, 42)\nappend(names, 'Alice')"
        },
        {
            name: 'pop(array)',
            category: 'array',
            description: 'Удаляет и возвращает последний элемент массива.',
            signature: 'pop(array)',
            example: "global last = pop(my_array)"
        },
        {
            name: 'sort(array)',
            category: 'array',
            description: 'Сортирует массив по возрастанию.',
            signature: 'sort(array)',
            example: "sort(numbers)\nsort(names)"
        },
        {
            name: 'unique(array)',
            category: 'array',
            description: 'Возвращает новый массив без дубликатов.',
            signature: 'unique(array)',
            example: "global unique_items = unique([1, 2, 2, 3, 3, 3])  # [1, 2, 3]"
        },
        {
            name: 'sum(array)',
            category: 'array',
            description: 'Вычисляет сумму чисел в массиве.',
            signature: 'sum(array)',
            example: "global total = sum([1, 2, 3, 4, 5])      # 15"
        },
        {
            name: 'average(array)',
            category: 'array',
            description: 'Вычисляет среднее значение чисел в массиве.',
            signature: 'average(array)',
            example: "global avg = average([1, 2, 3, 4, 5])    # 3"
        },
        {
            name: 'count(array)',
            category: 'array',
            description: 'Возвращает количество элементов в массиве.',
            signature: 'count(array)',
            example: "global items = count([1, 2, 3, 4, 5])    # 5"
        },
        {
            name: 'reverse(array)',
            category: 'array',
            description: 'Возвращает новый массив с элементами в обратном порядке.',
            signature: 'reverse(array)',
            example: "global reversed = reverse([1, 2, 3])  # [3, 2, 1]"
        },
        {
            name: 'range(start, end, step?)',
            category: 'array',
            description: 'Создает массив чисел от start до end с шагом step.',
            signature: 'range(start, end, [step])',
            example: "global numbers = range(1, 10)        # [1, 2, 3, ..., 9]\nglobal evens = range(0, 20, 2)       # [0, 2, 4, ..., 18]"
        }
    ],
    string: [
        {
            name: 'split(string, delimiter)',
            category: 'string',
            description: 'Разделяет строку на массив по разделителю.',
            signature: 'split(string, delimiter)',
            example: "global words = split('hello,world,datacode', ',')  # ['hello', 'world', 'datacode']"
        },
        {
            name: 'join(array, delimiter)',
            category: 'string',
            description: 'Объединяет массив строк в одну строку.',
            signature: 'join(array, delimiter)',
            example: "global text = join(['hello', 'world'], ' ')        # 'hello world'"
        },
        {
            name: 'trim(string)',
            category: 'string',
            description: 'Удаляет пробелы в начале и конце строки.',
            signature: 'trim(string)',
            example: "global clean = trim('  hello world  ')  # 'hello world'"
        },
        {
            name: 'upper(string)',
            category: 'string',
            description: 'Преобразует строку в верхний регистр.',
            signature: 'upper(string)',
            example: "global uppercase = upper('hello')       # 'HELLO'"
        },
        {
            name: 'lower(string)',
            category: 'string',
            description: 'Преобразует строку в нижний регистр.',
            signature: 'lower(string)',
            example: "global lowercase = lower('WORLD')       # 'world'"
        },
        {
            name: 'contains(string, substring)',
            category: 'string',
            description: 'Проверяет, содержит ли строка подстроку.',
            signature: 'contains(string, substring)',
            example: "global has_world = contains('hello world', 'world')  # true"
        }
    ],
    table: [
        {
            name: 'table(data, headers)',
            category: 'table',
            description: 'Создает таблицу из данных и заголовков.',
            signature: 'table(data, headers)',
            example: "global data = table([\n    ['Alice', 25, 'New York'],\n    ['Bob', 30, 'London']\n], ['Name', 'Age', 'City'])"
        },
        {
            name: 'show_table(table)',
            category: 'table',
            description: 'Отображает таблицу в отформатированном ASCII виде.',
            signature: 'show_table(table)',
            example: "show_table(my_table)"
        },
        {
            name: 'table_info(table)',
            category: 'table',
            description: 'Возвращает информацию о таблице (строки, столбцы, типы).',
            signature: 'table_info(table)',
            example: "global info = table_info(data)\nprint('Rows:', info.rows, 'Columns:', info.columns)"
        },
        {
            name: 'table_head(table, count?)',
            category: 'table',
            description: 'Возвращает первые N строк таблицы.',
            signature: 'table_head(table, [count])',
            example: "global first_10 = table_head(data, 10)\ntable_head(data, 5)"
        },
        {
            name: 'table_tail(table, count?)',
            category: 'table',
            description: 'Возвращает последние N строк таблицы.',
            signature: 'table_tail(table, [count])',
            example: "global last_5 = table_tail(data, 5)"
        },
        {
            name: 'table_headers(table)',
            category: 'table',
            description: 'Возвращает заголовки столбцов таблицы.',
            signature: 'table_headers(table)',
            example: "global headers = table_headers(data)\nprint('Columns:', headers)"
        },
        {
            name: 'table_select(table, columns)',
            category: 'table',
            description: 'Выбирает определенные столбцы из таблицы.',
            signature: 'table_select(table, columns)',
            example: "global subset = table_select(data, ['Name', 'Age'])"
        },
        {
            name: 'table_sort(table, column, ascending?)',
            category: 'table',
            description: 'Сортирует таблицу по указанному столбцу.',
            signature: 'table_sort(table, column, [ascending])',
            example: "global sorted_by_age = table_sort(data, 'Age', true)\nglobal sorted_by_name = table_sort(data, 'Name', false)"
        },
        {
            name: 'table_where(table, column, operator, value)',
            category: 'table',
            description: 'Фильтрует строки таблицы по условию.',
            signature: 'table_where(table, column, operator, value)',
            example: "global adults = table_where(data, 'Age', '>', 18)\nglobal ny_users = table_where(data, 'City', '==', 'New York')"
        },
        {
            name: 'table_filter(table, condition)',
            category: 'table',
            description: 'Фильтрует таблицу по строковому условию.',
            signature: 'table_filter(table, condition)',
            example: "global filtered = table_filter(data, 'Age > 25 AND City == \"New York\"')"
        },
        {
            name: 'table_distinct(table, column)',
            category: 'table',
            description: 'Возвращает уникальные значения из столбца.',
            signature: 'table_distinct(table, column)',
            example: "global cities = table_distinct(data, 'City')"
        },
        {
            name: 'table_join(left_table, right_table, left_key, right_key, join_type?)',
            category: 'table',
            description: 'Объединяет две таблицы по ключевым столбцам. Типы соединения: inner, left, right, outer.',
            signature: 'table_join(left_table, right_table, left_key, right_key, [join_type])',
            example: "global joined = table_join(users, orders, 'id', 'user_id', 'inner')\nglobal left_join = table_join(users, profiles, 'id', 'user_id', 'left')"
        },
        {
            name: 'table_union(table1, table2)',
            category: 'table',
            description: 'Объединяет строки двух таблиц с одинаковой структурой.',
            signature: 'table_union(table1, table2)',
            example: "global combined = table_union(data1, data2)"
        },
        {
            name: 'table_sample(table, count)',
            category: 'table',
            description: 'Возвращает случайную выборку строк из таблицы.',
            signature: 'table_sample(table, count)',
            example: "global sample = table_sample(large_dataset, 100)"
        },
        {
            name: 'enum(iterable)',
            category: 'table',
            description: 'Возвращает пары (индекс, значение) для итерации по массивам и таблицам.',
            signature: 'enum(iterable)',
            example: "for i, item in enum(my_array) do\n    print('Index:', i, 'Value:', item)\nforend"
        },
        {
            name: 'join(left, right, on, type?, suffixes?, nulls_equal?)',
            category: 'table',
            description: 'Универсальная функция объединения таблиц. Типы: inner, left, right, full, cross, semi, anti.',
            signature: 'join(left, right, on, [type], [suffixes], [nulls_equal])',
            example: "global users = table([[1, 'Alice'], [2, 'Bob']], ['id', 'name'])\nglobal orders = table([[1, 100], [1, 200]], ['user_id', 'amount'])\nglobal result = join(users, orders, 'id', 'user_id', 'inner')"
        },
        {
            name: 'inner_join(left, right, on)',
            category: 'table',
            description: 'Внутреннее соединение таблиц. Возвращает строки, для которых существует совпадение в обеих таблицах.',
            signature: 'inner_join(left, right, on)',
            example: "global users = table([[1, 'Alice'], [2, 'Bob']], ['id', 'name'])\nglobal orders = table([[1, 100]], ['user_id', 'amount'])\nglobal result = inner_join(users, orders, 'id', 'user_id')"
        },
        {
            name: 'left_join(left, right, on)',
            category: 'table',
            description: 'Левое соединение таблиц. Все строки из left сохраняются, отсутствующие значения из right заполняются NULL.',
            signature: 'left_join(left, right, on)',
            example: "global users = table([[1, 'Alice'], [2, 'Bob']], ['id', 'name'])\nglobal orders = table([[1, 100]], ['user_id', 'amount'])\nglobal result = left_join(users, orders, 'id', 'user_id')"
        },
        {
            name: 'right_join(left, right, on)',
            category: 'table',
            description: 'Правое соединение таблиц. Все строки из right сохраняются, отсутствующие значения из left заполняются NULL.',
            signature: 'right_join(left, right, on)',
            example: "global users = table([[1, 'Alice'], [2, 'Bob']], ['id', 'name'])\nglobal orders = table([[1, 100], [3, 200]], ['user_id', 'amount'])\nglobal result = right_join(users, orders, 'id', 'user_id')"
        },
        {
            name: 'full_join(left, right, on)',
            category: 'table',
            description: 'Полное внешнее соединение таблиц. Возвращает все строки из обеих таблиц, отсутствующие значения заполняются NULL.',
            signature: 'full_join(left, right, on)',
            example: "global users = table([[1, 'Alice'], [2, 'Bob']], ['id', 'name'])\nglobal orders = table([[1, 100], [3, 200]], ['user_id', 'amount'])\nglobal result = full_join(users, orders, 'id', 'user_id')"
        },
        {
            name: 'cross_join(left, right)',
            category: 'table',
            description: 'Декартово произведение таблиц. Возвращает все комбинации строк из обеих таблиц.',
            signature: 'cross_join(left, right)',
            example: "global table1 = table([[1], [2]], ['col1'])\nglobal table2 = table([['a'], ['b']], ['col2'])\nglobal result = cross_join(table1, table2)  # 4 строки"
        },
        {
            name: 'semi_join(left, right, on)',
            category: 'table',
            description: 'Полу-соединение таблиц. Возвращает только строки left, имеющие совпадения в right, без колонок right.',
            signature: 'semi_join(left, right, on)',
            example: "global users = table([[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']], ['id', 'name'])\nglobal orders = table([[1, 100], [3, 300]], ['user_id', 'amount'])\nglobal result = semi_join(users, orders, 'id', 'user_id')"
        },
        {
            name: 'anti_join(left, right, on)',
            category: 'table',
            description: 'Анти-соединение таблиц. Возвращает строки left, не имеющие совпадений в right.',
            signature: 'anti_join(left, right, on)',
            example: "global users = table([[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']], ['id', 'name'])\nglobal orders = table([[1, 100], [3, 300]], ['user_id', 'amount'])\nglobal result = anti_join(users, orders, 'id', 'user_id')  # только Bob"
        },
        {
            name: 'asof_join(left, right, on, direction?)',
            category: 'table',
            description: 'Временное соединение таблиц (ASOF JOIN). Объединяет таблицы по временной близости значений.',
            signature: 'asof_join(left, right, on, [direction])',
            example: "global prices = table([['2024-01-01', 100], ['2024-01-05', 105]], ['date', 'price'])\nglobal trades = table([['2024-01-02', 'buy'], ['2024-01-06', 'sell']], ['date', 'action'])\nglobal result = asof_join(trades, prices, 'date', 'date', 'backward')"
        }
    ],
    ml: [
        {
            name: 'ml.tensor(data, shape?)',
            category: 'ml',
            description: 'Создает тензор из данных. Форма может быть автоматически определена из структуры данных.',
            signature: 'ml.tensor(data, [shape])',
            example: "import ml\nglobal t1 = ml.tensor([1.0, 2.0, 3.0])  # Форма: [3]\nglobal t2 = ml.tensor([[1, 2], [3, 4]])  # Форма: [2, 2]"
        },
        {
            name: 'ml.shape(tensor)',
            category: 'ml',
            description: 'Возвращает форму тензора.',
            signature: 'ml.shape(tensor)',
            example: "import ml\nglobal t = ml.tensor([[1, 2], [3, 4]])\nglobal shape = ml.shape(t)  # [2, 2]"
        },
        {
            name: 'ml.data(tensor)',
            category: 'ml',
            description: 'Возвращает данные тензора как плоский массив.',
            signature: 'ml.data(tensor)',
            example: "import ml\nglobal t = ml.tensor([[1, 2], [3, 4]])\nglobal data = ml.data(t)  # [1.0, 2.0, 3.0, 4.0]"
        },
        {
            name: 'ml.add(t1, t2)',
            category: 'ml',
            description: 'Поэлементное сложение двух тензоров.',
            signature: 'ml.add(t1, t2)',
            example: "import ml\nglobal t1 = ml.tensor([1, 2, 3])\nglobal t2 = ml.tensor([4, 5, 6])\nglobal result = ml.add(t1, t2)  # [5, 7, 9]"
        },
        {
            name: 'ml.sub(t1, t2)',
            category: 'ml',
            description: 'Поэлементное вычитание двух тензоров.',
            signature: 'ml.sub(t1, t2)',
            example: "import ml\nglobal t1 = ml.tensor([5, 7, 9])\nglobal t2 = ml.tensor([1, 2, 3])\nglobal result = ml.sub(t1, t2)  # [4, 5, 6]"
        },
        {
            name: 'ml.mul(t1, t2)',
            category: 'ml',
            description: 'Поэлементное умножение двух тензоров.',
            signature: 'ml.mul(t1, t2)',
            example: "import ml\nglobal t1 = ml.tensor([2, 3, 4])\nglobal t2 = ml.tensor([5, 6, 7])\nglobal result = ml.mul(t1, t2)  # [10, 18, 28]"
        },
        {
            name: 'ml.matmul(t1, t2)',
            category: 'ml',
            description: 'Матричное умножение двух тензоров.',
            signature: 'ml.matmul(t1, t2)',
            example: "import ml\nglobal t1 = ml.tensor([[1, 2], [3, 4]])\nglobal t2 = ml.tensor([[5, 6], [7, 8]])\nglobal result = ml.matmul(t1, t2)  # [[19, 22], [43, 50]]"
        },
        {
            name: 'ml.neural_network(layers)',
            category: 'ml',
            description: 'Создает нейронную сеть из последовательности слоев.',
            signature: 'ml.neural_network(layers)',
            example: "import ml\nglobal layer1 = ml.layer.linear(784, 128)\nglobal layer2 = ml.layer.relu()\nglobal layer3 = ml.layer.linear(128, 10)\nglobal model = ml.neural_network(ml.sequential([layer1, layer2, layer3]))"
        },
        {
            name: 'ml.layer.linear(input_size, output_size)',
            category: 'ml',
            description: 'Создает линейный (полносвязный) слой нейронной сети.',
            signature: 'ml.layer.linear(input_size, output_size)',
            example: "import ml\nglobal layer = ml.layer.linear(784, 128)  # Слой с 784 входами и 128 выходами"
        },
        {
            name: 'ml.layer.relu()',
            category: 'ml',
            description: 'Создает слой активации ReLU (Rectified Linear Unit).',
            signature: 'ml.layer.relu()',
            example: "import ml\nglobal activation = ml.layer.relu()"
        },
        {
            name: 'ml.layer.softmax()',
            category: 'ml',
            description: 'Создает слой активации Softmax для многоклассовой классификации.',
            signature: 'ml.layer.softmax()',
            example: "import ml\nglobal softmax = ml.layer.softmax()"
        },
        {
            name: 'ml.layer.flatten()',
            category: 'ml',
            description: 'Создает слой Flatten для преобразования многомерных тензоров в одномерные.',
            signature: 'ml.layer.flatten()',
            example: "import ml\nglobal flatten = ml.layer.flatten()"
        },
        {
            name: 'model.train(x_train, y_train, epochs, batch_size, learning_rate, loss_fn)',
            category: 'ml',
            description: 'Обучает нейронную сеть. Возвращает историю потерь.',
            signature: 'model.train(x_train, y_train, epochs, batch_size, learning_rate, loss_fn)',
            example: "import ml\nglobal model = ml.neural_network(...)\nglobal loss_history = model.train(x_train, y_train, 10, 32, 0.001, 'cross_entropy')"
        },
        {
            name: 'model.predict(x)',
            category: 'ml',
            description: 'Выполняет предсказание на входных данных.',
            signature: 'model.predict(x)',
            example: "import ml\nglobal predictions = model.predict(x_test)"
        },
        {
            name: 'model.save(path)',
            category: 'ml',
            description: 'Сохраняет модель в файл.',
            signature: 'model.save(path)',
            example: "import ml\nglobal model = ml.neural_network(...)\nmodel.save(path('model.nn'))"
        },
        {
            name: 'ml.load_model(path)',
            category: 'ml',
            description: 'Загружает модель из файла.',
            signature: 'ml.load_model(path)',
            example: "import ml\nglobal model = ml.load_model(path('model.nn'))"
        },
        {
            name: 'ml.optimizer.sgd(learning_rate)',
            category: 'ml',
            description: 'Создает оптимизатор SGD (Stochastic Gradient Descent).',
            signature: 'ml.optimizer.sgd(learning_rate)',
            example: "import ml\nglobal optimizer = ml.optimizer.sgd(0.01)"
        },
        {
            name: 'ml.optimizer.adam(learning_rate)',
            category: 'ml',
            description: 'Создает оптимизатор Adam (Adaptive Moment Estimation).',
            signature: 'ml.optimizer.adam(learning_rate)',
            example: "import ml\nglobal optimizer = ml.optimizer.adam(0.001)"
        },
        {
            name: 'ml.loss.mse(y_true, y_pred)',
            category: 'ml',
            description: 'Вычисляет функцию потерь MSE (Mean Squared Error).',
            signature: 'ml.loss.mse(y_true, y_pred)',
            example: "import ml\nglobal loss = ml.loss.mse(y_true, y_pred)"
        },
        {
            name: 'ml.loss.cross_entropy(y_true, y_pred)',
            category: 'ml',
            description: 'Вычисляет функцию потерь Cross Entropy для классификации.',
            signature: 'ml.loss.cross_entropy(y_true, y_pred)',
            example: "import ml\nglobal loss = ml.loss.cross_entropy(y_true, y_pred)"
        },
        {
            name: 'ml.dataset.load_mnist()',
            category: 'ml',
            description: 'Загружает датасет MNIST для обучения нейронных сетей.',
            signature: 'ml.dataset.load_mnist()',
            example: "import ml\nglobal x_train, y_train, x_test, y_test = ml.dataset.load_mnist()"
        }
    ]
};

function loadFunctions() {
    const functionsGrid = document.getElementById('functions-grid');
    if (!functionsGrid) return;

    // Clear existing functions to avoid duplicates
    functionsGrid.innerHTML = '';

    // Flatten all functions
    const allFunctions = Object.values(functionsData).flat();

    allFunctions.forEach(func => {
        const card = document.createElement('div');
        card.className = 'function-card';
        card.setAttribute('data-category', func.category);

        card.innerHTML = `
            <div class="function-name">${func.name}</div>
            <span class="function-category">${func.category}</span>
            <div class="function-description">${func.description}</div>
            <div class="function-signature">${func.signature}</div>
            <div class="function-example" style="position: relative;">
                <button class="function-example-btn" onclick="copyExample(this)">📋</button>
                <pre><code>${func.example}</code></pre>
            </div>
        `;

        functionsGrid.appendChild(card);
    });

    // Setup filtering after functions are loaded
    setupFunctionFiltering();
}

function setupFunctionFiltering() {
    const categoryButtons = document.querySelectorAll('.category-btn');
    
    if (categoryButtons.length === 0) {
        return;
    }
    
    categoryButtons.forEach(button => {
        // Remove existing event listeners by removing and re-adding
        const newButton = button.cloneNode(true);
        button.replaceWith(newButton);
        
        newButton.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const category = newButton.getAttribute('data-category');
            
            // Remove active class from all category buttons
            document.querySelectorAll('.category-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            newButton.classList.add('active');

            // Get all function cards (they should exist now)
            const functionCards = document.querySelectorAll('.function-card');
            
            // Filter function cards
            functionCards.forEach(card => {
                const cardCategory = card.getAttribute('data-category');
                if (category === 'all' || cardCategory === category) {
                    card.classList.remove('hidden');
                    // Add fade-in animation
                    card.style.opacity = '0';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transition = 'opacity 0.3s ease';
                    }, 10);
                } else {
                    card.classList.add('hidden');
                }
            });
        });
    });
}

function copyExample(button) {
    const exampleBlock = button.closest('.function-example');
    const code = exampleBlock.querySelector('code').textContent;

    navigator.clipboard.writeText(code).then(() => {
        const originalText = button.textContent;
        button.textContent = '✓';
        button.style.background = '#10b981';

        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy code:', err);
        button.textContent = '❌';
        setTimeout(() => {
            button.textContent = '📋';
        }, 2000);
    });
}
