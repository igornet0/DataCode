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


function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function loadFunctions() {
    const functionsGrid = document.getElementById('functions-grid');
    const categoriesEl = document.getElementById('function-categories');
    if (!functionsGrid) return;

    const lang = window.DATACODE_LANG;
    if (!lang || !lang.builtins) {
        functionsGrid.innerHTML = '<p>Не удалось загрузить каталог функций (datacode_lang.js).</p>';
        return;
    }

    const categoryOrder = ['utility', 'type', 'datetime', 'path', 'math', 'string', 'array', 'table', 'join', 'crypto'];
    const names = lang.categoryNames || {};

    if (categoriesEl) {
        const buttons = ['<button class="category-btn active" data-category="all">Все функции</button>'];
        categoryOrder.forEach((id) => {
            buttons.push(`<button class="category-btn" data-category="${id}">${escapeHtml(names[id] || id)}</button>`);
        });
        categoriesEl.innerHTML = buttons.join('');
    }

    functionsGrid.innerHTML = '';
    lang.builtins.forEach((func) => {
        const card = document.createElement('div');
        card.className = 'function-card';
        card.setAttribute('data-category', func.category);
        const categoryLabel = names[func.category] || func.category;
        const example = func.example ? `<div class="function-example" style="position: relative;">
                <button class="function-example-btn" onclick="copyExample(this)">Копировать</button>
                <pre><code>${escapeHtml(func.example)}</code></pre>
            </div>` : '';
        card.innerHTML = `
            <div class="function-name">${escapeHtml(func.name)}</div>
            <span class="function-category">${escapeHtml(categoryLabel)}</span>
            <div class="function-description">${escapeHtml(func.description)}</div>
            <div class="function-signature">${escapeHtml(func.signature)}</div>
            ${example}
        `;
        functionsGrid.appendChild(card);
    });

    setupFunctionFiltering();
}

function setupFunctionFiltering() {
    const categoryButtons = document.querySelectorAll('.category-btn');
    if (categoryButtons.length === 0) return;

    categoryButtons.forEach((button) => {
        const newButton = button.cloneNode(true);
        button.replaceWith(newButton);

        newButton.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            const category = newButton.getAttribute('data-category');

            document.querySelectorAll('.category-btn').forEach((btn) => btn.classList.remove('active'));
            newButton.classList.add('active');

            document.querySelectorAll('.function-card').forEach((card) => {
                const cardCategory = card.getAttribute('data-category');
                if (category === 'all' || cardCategory === category) {
                    card.classList.remove('hidden');
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
    }).catch(() => {
        button.textContent = '✗';
        setTimeout(() => {
            button.textContent = 'Копировать';
        }, 2000);
    });
}
