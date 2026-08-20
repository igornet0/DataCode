/**
 * DataCode Syntax Highlighter
 * Keywords, builtins and type methods come from DATACODE_LANG.
 */

class DataCodeSyntaxHighlighter {
    constructor() {
        const lang = (typeof DATACODE_LANG !== 'undefined') ? DATACODE_LANG : null;

        this.keywords = lang ? lang.keywords.slice() : [
            'global', 'let', 'fn', 'stream', 'if', 'else', 'for', 'in', 'while',
            'return', 'ireturn', 'ereturn', 'try', 'catch', 'finally', 'throw',
            'break', 'continue', 'import', 'from', 'as', 'cls', 'new', 'super',
            'this', 'Abstract', 'public', 'private', 'protected',
            'true', 'false', 'null', 'and', 'or', 'inf', 'nan'
        ];

        this.keywordSet = new Set(this.keywords.map((k) => k === 'Abstract' ? k : k.toLowerCase()));

        this.builtinFunctions = lang ? lang.builtinNames.slice() : [];
        this.builtinSet = new Set(this.builtinFunctions.map((n) => n.toLowerCase()));

        this.operators = (lang ? lang.operators.slice() : [
            '==', '!=', '<=', '>=', '<<', '>>', 'and', 'or', 'in',
            '+', '-', '*', '/', '%', '<', '>', '!', '&', '|', '^', '~', '?', ':', '='
        ]).slice().sort((a, b) => b.length - a.length);

        this.functionDefinitions = {};
        if (lang && lang.builtins) {
            lang.builtins.forEach((def) => {
                this.functionDefinitions[def.name.toLowerCase()] = def;
            });
        }

        this.lang = lang;
        this.allMethodNames = new Set();
        if (lang && lang.typeMethods) {
            Object.keys(lang.typeMethods).forEach((typeName) => {
                lang.typeMethods[typeName].forEach((m) => this.allMethodNames.add(m.name.toLowerCase()));
            });
        }
    }

    isKeyword(ident) {
        if (ident === 'Abstract') return true;
        if (ident.toLowerCase() === 'abstract') return false;
        return this.keywordSet.has(ident.toLowerCase());
    }

    isBuiltin(ident) {
        return this.builtinSet.has(ident.toLowerCase());
    }

    getFunctionDefinition(name) {
        if (!name) return null;
        return this.functionDefinitions[String(name).toLowerCase()] || null;
    }

    highlight(code) {
        if (!code) return '';
        const lines = code.split('\n');
        return lines.map((line) => this.highlightLine(line)).join('\n');
    }

    highlightLine(line) {
        if (!line) return '';

        let html = '';
        let i = 0;
        const len = line.length;
        let prevWasDot = false;

        while (i < len) {
            if (line[i] === '#') {
                html += `<span class="comment">${this.escapeHtml(line.substring(i))}</span>`;
                break;
            }

            if (line[i] === "'" && (i === 0 || line[i - 1] !== '\\')) {
                const result = this.consumeString(line, i, "'");
                html += `<span class="string">${this.escapeHtml(result.text)}</span>`;
                i = result.end;
                prevWasDot = false;
                continue;
            }

            if (line[i] === '"' && (i === 0 || line[i - 1] !== '\\')) {
                html += this.highlightDoubleString(line, i);
                i = this.consumeString(line, i, '"').end;
                prevWasDot = false;
                continue;
            }

            if (this.isDigit(line[i]) || (line[i] === '-' && i + 1 < len && this.isDigit(line[i + 1]) && (i === 0 || !this.isIdentifierChar(line[i - 1])))) {
                if (i > 0 && this.isIdentifierChar(line[i - 1])) {
                    html += this.escapeHtml(line[i]);
                    i++;
                    continue;
                }
                let numEnd = i + 1;
                let hasDot = false;
                while (numEnd < len) {
                    if (this.isDigit(line[numEnd])) {
                        numEnd++;
                    } else if (line[numEnd] === '.' && !hasDot && numEnd + 1 < len && this.isDigit(line[numEnd + 1])) {
                        hasDot = true;
                        numEnd++;
                    } else {
                        break;
                    }
                }
                html += `<span class="number">${this.escapeHtml(line.substring(i, numEnd))}</span>`;
                i = numEnd;
                prevWasDot = false;
                continue;
            }

            if (this.isIdentifierStart(line[i])) {
                let identEnd = i + 1;
                while (identEnd < len && this.isIdentifierChar(line[identEnd])) {
                    identEnd++;
                }
                const ident = line.substring(i, identEnd);
                const escaped = this.escapeHtml(ident);

                if (this.isKeyword(ident)) {
                    html += `<span class="keyword">${escaped}</span>`;
                } else if (this.isBuiltin(ident)) {
                    html += `<span class="builtin" data-function="${this.escapeHtml(ident.toLowerCase())}">${escaped}</span>`;
                } else if (prevWasDot) {
                    html += `<span class="method" data-method="${escaped}">${escaped}</span>`;
                } else if (identEnd < len && line[identEnd] === '(') {
                    html += `<span class="function">${escaped}</span>`;
                } else {
                    html += `<span class="variable">${escaped}</span>`;
                }

                i = identEnd;
                prevWasDot = false;
                continue;
            }

            if (line[i] === '{' || line[i] === '}') {
                html += `<span class="operator">${this.escapeHtml(line[i])}</span>`;
                i++;
                prevWasDot = false;
                continue;
            }

            if (line[i] === '.') {
                html += `<span class="operator">.</span>`;
                i++;
                prevWasDot = true;
                continue;
            }

            let operatorMatched = false;
            for (const op of this.operators) {
                if (line.substring(i, i + op.length) !== op) continue;
                const beforeId = i > 0 && this.isIdentifierChar(line[i - 1]);
                const afterId = i + op.length < len && this.isIdentifierChar(line[i + op.length]);
                if ((op === 'and' || op === 'or' || op === 'in') && (beforeId || afterId)) {
                    continue;
                }
                html += `<span class="operator">${this.escapeHtml(op)}</span>`;
                i += op.length;
                operatorMatched = true;
                prevWasDot = false;
                break;
            }

            if (operatorMatched) continue;

            html += this.escapeHtml(line[i]);
            prevWasDot = false;
            i++;
        }

        return html;
    }

    consumeString(line, start, quote) {
        let end = start + 1;
        while (end < line.length) {
            if (line[end] === '\\') {
                end += 2;
                continue;
            }
            if (line[end] === quote) {
                end++;
                break;
            }
            end++;
        }
        return { text: line.substring(start, end), end: end };
    }

    highlightDoubleString(line, start) {
        const consumed = this.consumeString(line, start, '"');
        const raw = consumed.text;
        let inner = raw.slice(1, raw.endsWith('"') && raw.length > 1 ? -1 : undefined);
        let out = '<span class="string">"</span>';
        let i = 0;
        while (i < inner.length) {
            if (inner[i] === '\\' && i + 1 < inner.length) {
                out += `<span class="string">${this.escapeHtml(inner.substring(i, i + 2))}</span>`;
                i += 2;
                continue;
            }
            if (inner[i] === '$' && inner[i + 1] === '{') {
                const close = inner.indexOf('}', i + 2);
                if (close !== -1) {
                    out += `<span class="interpolation">${this.escapeHtml(inner.substring(i, close + 1))}</span>`;
                    i = close + 1;
                    continue;
                }
            }
            let next = inner.indexOf('${', i);
            if (next === -1) next = inner.length;
            if (next > i) {
                out += `<span class="string">${this.escapeHtml(inner.substring(i, next))}</span>`;
                i = next;
            } else {
                out += `<span class="string">${this.escapeHtml(inner[i])}</span>`;
                i++;
            }
        }
        if (raw.endsWith('"') && raw.length > 1) {
            out += '<span class="string">"</span>';
        }
        return out;
    }

    isDigit(ch) {
        return ch >= '0' && ch <= '9';
    }

    isIdentifierStart(ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_';
    }

    isIdentifierChar(ch) {
        return this.isIdentifierStart(ch) || this.isDigit(ch);
    }

    escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    getTokens(code) {
        const tokens = new Set();
        const regex = /\b([A-Za-z_][A-Za-z0-9_]*)\b/g;
        let match;
        while ((match = regex.exec(code)) !== null) {
            const token = match[1];
            if (!this.isKeyword(token) && !this.isBuiltin(token)) {
                tokens.add(token);
            }
        }
        return Array.from(tokens);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataCodeSyntaxHighlighter;
}
