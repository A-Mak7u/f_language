#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
полный учебниковый лексический анализатор на конечных автоматах (ε-nfa → maximal munch) с трассировкой.
всё в одном файле для удобства вставки в отчёт.

язык (минимальная грамматика под лабораторную):
  - ключевые слова: do, while, loop
  - идентификаторы: letter (letter|digit)*
  - числа:          [+-]? digit+       (диапазон контролируем как [-32768..32767] после распознавания)
  - операторы:      =, ==, <, >, <=, >=, <>, +, -, *, /
  - разделители:    ;, (, )
  - пробелы:        разделяют токены, сами токен не образуют
  - комментарии:    // до конца строки (игнорируются)

возможности:
  - строит отдельные ε-nfa для каждой категории токенов и объединяет их в один большой ε-nfa;
  - печатает δ-таблицы каждого под-автомата и объединённого автомата;
  - симулирует распознавание входа maximal munch + приоритет категорий;
  - для каждого токена печатает пошаговую трассировку (множества состояний до/после шага);
  - сохраняет .dot (всегда) и .png (если установлен python-graphviz + system graphviz).
"""

import enum
import sys
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Callable, Iterable
from tabulate import tabulate

# попытка подключить graphviz для генерации png; если не получится — не падаем
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except Exception:
    HAS_GRAPHVIZ = False


# =========================================
# базовые типы: категории токенов, структура токена
# =========================================

class TokenCat(enum.Enum):
    KEYWORD    = "KEYWORD"
    IDENT      = "IDENTIFIER"
    NUMBER     = "CONSTANT"
    OPERATOR   = "OPERATOR"
    SEPARATOR  = "SEPARATOR"


@dataclass
class Token:
    category: TokenCat
    value: str


# набор ключевых слов языка (для демонстрации)
KEYWORDS = {"do", "while", "loop"}

# приоритет категорий при равной длине (для maximal munch с предписанным разбором)
CATEGORY_PRIORITY = {
    TokenCat.KEYWORD:   5,
    TokenCat.IDENT:     4,
    TokenCat.NUMBER:    3,
    TokenCat.OPERATOR:  2,
    TokenCat.SEPARATOR: 1,
}

# предельная длина идентификатора (задание часто упоминает)
IDENT_MAXLEN = 255

# допустимый диапазон для целых
INT_MIN, INT_MAX = -32768, 32767


# =========================================
# ε-nfa: формальная модель и базовые операции
# =========================================

class ENFA:
    """
    ε-nfa: мы работаем с множествами состояний, δ и ε-переходами.
    у рёбер есть два вида меток:
      - точные символы (строка длиной 1),
      - предикаты (callable(ch)->bool) для классов символов; у предиката есть печатное имя.

    accept_info[state] = (category, tag) — помечаем принимающие состояния и их семантику.
    tag используется для человекочитаемости (например, конкретное ключевое слово или имя класса).
    """

    def __init__(self):
        self.next_state_id: int = 0
        self.start: Optional[int] = None
        self.states: Set[int] = set()

        # δ: переходы по точным символам и ε
        self.delta: Dict[int, Dict[str, List[int]]] = {}                 # state -> label -> [state]
        # переходы по предикатам (классам символов)
        self.pred_delta: Dict[int, List[Tuple[Callable[[str], bool], str, int]]] = {}  # state -> [(pred, name, next)]

        # принимающие состояния
        self.accept_info: Dict[int, Tuple[TokenCat, Optional[str]]] = {}

    # ---------- выделение нового состояния ----------
    def _new_state(self) -> int:
        s = self.next_state_id
        self.next_state_id += 1
        self.states.add(s)
        return s

    # ---------- добавление рёбер ----------
    def _add_edge(self, s_from: int, label: str, s_to: int):
        """
        добавляет ребро s_from --label--> s_to; label может быть 'ε' или конкретным символом
        """
        self.delta.setdefault(s_from, {}).setdefault(label, []).append(s_to)

    def _add_pred_edge(self, s_from: int, pred: Callable[[str], bool], pred_name: str, s_to: int):
        """
        добавляет ребро s_from --<pred_name>--> s_to; предикат проверяется на каждом символе
        """
        self.pred_delta.setdefault(s_from, []).append((pred, pred_name, s_to))

    # ---------- генераторы простых nfa ----------
    def add_string_nfa(self, s: str, category: TokenCat, tag: Optional[str] = None) -> Tuple[int, int]:
        """
        строит линейный nfa, принимающий ровно строку s.
        возвращает (start, accept).
        """
        cur = self._new_state()
        start = cur
        for ch in s:
            nxt = self._new_state()
            self._add_edge(cur, ch, nxt)
            cur = nxt
        self.accept_info[cur] = (category, tag if tag is not None else s)
        return start, cur

    # ---------- маркировка принимающих ----------
    def mark_accept(self, state: int, category: TokenCat, tag: Optional[str] = None):
        self.accept_info[state] = (category, tag)

    # ---------- ε-замыкание ----------
    def eps_closure(self, S: Set[int]) -> Set[int]:
        """
        возвращает ε-замыкание множества состояний S.
        """
        st = set(S)
        stack = list(S)
        while stack:
            q = stack.pop()
            for nxt in self.delta.get(q, {}).get("ε", []):
                if nxt not in st:
                    st.add(nxt)
                    stack.append(nxt)
        return st

    # ---------- шаг по одному символу ----------
    def move(self, S: Set[int], ch: str) -> Set[int]:
        """
        множество состояний, достижимых из S по символу ch (без ε-замыкания).
        учитывает и точные метки, и предикаты.
        """
        out = set()
        for q in S:
            # точные метки
            for nxt in self.delta.get(q, {}).get(ch, []):
                out.add(nxt)
            # предикаты
            for pred, _name, nxt in self.pred_delta.get(q, []):
                if pred(ch):
                    out.add(nxt)
        return out

    # ---------- удобная печать δ-таблицы ----------
    def print_delta_table(self, title="δ-таблица", show_predicates=True):
        # собираем все встречающиеся точные метки (кроме ε)
        labels = set()
        for m in self.delta.values():
            for lab in m:
                if lab != "ε":
                    labels.add(lab)
        labels = sorted(labels)

        # собираем имена предикатов (классов символов)
        pred_names = set()
        if show_predicates:
            for plist in self.pred_delta.values():
                for _pred, name, _ in plist:
                    pred_names.add(f"<{name}>")
        pred_names = sorted(pred_names)

        headers = ["состояние"] + labels + pred_names + ["ε"]

        # декоратор для красивой маркировки начального/принимающего
        def deco(name: str, is_start: bool, is_acc: bool):
            if is_start: name = "→" + name
            if is_acc: name = "*" + name
            return name

        rows: List[List[str]] = []
        for q in sorted(self.states):
            row = [deco(f"q{q}", self.start == q, q in self.accept_info)]
            # точные метки
            for lab in labels:
                dest = self.delta.get(q, {}).get(lab, [])
                row.append("{" + ",".join(f"q{t}" for t in dest) + "}" if dest else "-")
            # предикатные
            for pname in pred_names:
                pname_raw = pname.strip("<>")
                dests: List[int] = []
                for (pred, name, t) in self.pred_delta.get(q, []):
                    if name == pname_raw:
                        dests.append(t)
                row.append("{" + ",".join(f"q{t}" for t in dests) + "}" if dests else "-")
            # ε
            eps = self.delta.get(q, {}).get("ε", [])
            row.append("{" + ",".join(f"q{t}" for t in eps) + "}" if eps else "-")
            rows.append(row)

        print(f"\n{title}")
        print(tabulate(rows, headers=headers, tablefmt="github",
                       colalign=("left", *("center",)*(len(headers)-1))))


# =========================================
# предикаты (классы символов)
# =========================================

def is_letter(ch: str) -> bool:
    # можно расширить при необходимости (поддержка '_', русских букв и т.д.)
    return ch.isalpha()

def is_digit(ch: str) -> bool:
    return ch.isdigit()

def is_sign(ch: str) -> bool:
    return ch in "+-"


# =========================================
# генераторы под-автоматов токенов
# =========================================

def build_keyword_nfa(word: str) -> ENFA:
    """
    ключевое слово как линейный nfa; один accept на последний символ
    """
    e = ENFA()
    s, a = e.add_string_nfa(word, TokenCat.KEYWORD, tag=word)
    e.start = s
    return e

def build_identifier_nfa() -> ENFA:
    """
    идентификатор: letter (letter|digit)*
    акцептор — любое состояние после чтения >=1 символа.
    """
    e = ENFA()
    q0 = e._new_state(); e.start = q0
    q1 = e._new_state()
    e._add_pred_edge(q0, is_letter, "LETTER", q1)
    # петля по letter|digit
    e._add_pred_edge(q1, is_letter, "LETTER", q1)
    e._add_pred_edge(q1, is_digit,  "DIGIT",  q1)
    e.mark_accept(q1, TokenCat.IDENT, "ID")
    return e

def build_number_nfa() -> ENFA:
    """
    число: [+-]? digit+
    акцептор — состояние после чтения 1+ цифр (знак опционален)
    """
    e = ENFA()
    q0 = e._new_state(); e.start = q0

    q_sign = e._new_state()
    e._add_pred_edge(q0, is_sign, "SIGN", q_sign)

    q1 = e._new_state()
    e._add_pred_edge(q0,     is_digit, "DIGIT", q1)
    e._add_pred_edge(q_sign, is_digit, "DIGIT", q1)

    # петля по цифрам
    e._add_pred_edge(q1, is_digit, "DIGIT", q1)

    e.mark_accept(q1, TokenCat.NUMBER, "INT")
    return e

def build_operator_nfa_list() -> ENFA:
    """
    операторы: объединение линейных nfa для каждого символа/двусимвольной последовательности.
    """
    ops = ["==", "<=", ">=", "<>", "=", "<", ">", "+", "-", "*", "/"]
    e = ENFA()
    starts: List[Tuple[int, int]] = []
    for op in ops:
        s, a = e.add_string_nfa(op, TokenCat.OPERATOR, tag=op)
        starts.append((s, a))
    # объединяем через новый старт и ε-переходы
    s0 = e._new_state(); e.start = s0
    for s, _a in starts:
        e._add_edge(s0, "ε", s)
    return e

def build_separator_nfa_list() -> ENFA:
    """
    разделители: ; ( )
    """
    seps = [";", "(", ")"]
    e = ENFA()
    s0 = e._new_state(); e.start = s0
    for sp in seps:
        s, _a = e.add_string_nfa(sp, TokenCat.SEPARATOR, tag=sp)
        e._add_edge(s0, "ε", s)
    return e


# =========================================
# объединение всех под-автоматов в один ε-nfa
# =========================================

def union_enfas(enfas: List[ENFA]) -> ENFA:
    """
    делает «большой» ε-nfa: новый старт —ε→ старты всех под-автоматов.
    при копировании под-автоматов состояния перенумеровываются (чтобы не было конфликтов id).
    """
    big = ENFA()

    def copy_enfa(src: ENFA) -> Dict[int, int]:
        """
        копирует src в big, возвращает отображение старых id в новые.
        """
        mapping: Dict[int, int] = {}
        # сначала создаём все состояния
        for s in sorted(src.states):
            ns = big._new_state()
            mapping[s] = ns
        # копируем δ
        for s in src.states:
            for lab, lst in src.delta.get(s, {}).items():
                for t in lst:
                    big._add_edge(mapping[s], lab, mapping[t])
            for (pred, name, t) in src.pred_delta.get(s, []):
                big._add_pred_edge(mapping[s], pred, name, mapping[t])
        # перенести принимающие
        for s, info in src.accept_info.items():
            big.accept_info[mapping[s]] = info
        return mapping

    maps = [copy_enfa(e) for e in enfas]
    s0 = big._new_state(); big.start = s0
    for e, m in zip(enfas, maps):
        big._add_edge(s0, "ε", m[e.start])
    return big


# =========================================
# трассировка одного прогона и maximal munch
# =========================================

@dataclass
class StepTrace:
    pos: int            # позиция читаемого символа в исходной строке
    char: str           # сам символ (repr)
    before: str         # множество состояний до чтения (как строка)
    after: str          # множество состояний после шага + ε-замыкания (как строка)

@dataclass
class AcceptHit:
    end_index: int
    category: TokenCat
    lexeme: str
    tag: Optional[str]
    priority: int

def fmt_states(S: Set[int]) -> str:
    return "{" + ",".join(sorted(f"q{x}" for x in S)) + "}" if S else "∅"

def run_maximal_munch_with_trace(big: ENFA, text: str, start_pos: int) -> Tuple[Optional[AcceptHit], List[StepTrace]]:
    """
    симулирует ε-nfa начиная с позиции start_pos, возвращает лучшее принятие (по длине+приоритету) и трассировку.
    """
    trace: List[StepTrace] = []
    i = start_pos
    cur = big.eps_closure({big.start})

    def record_accept(until_index: int, cur_states: Set[int], best: Optional[AcceptHit]) -> Optional[AcceptHit]:
        for st in cur_states:
            if st in big.accept_info:
                cat, tag = big.accept_info[st]
                lex = text[start_pos:until_index]
                cand = AcceptHit(until_index, cat, lex, tag, CATEGORY_PRIORITY.get(cat, 0))
                if best is None or len(cand.lexeme) > len(best.lexeme) or \
                   (len(cand.lexeme) == len(best.lexeme) and cand.priority > best.priority):
                    best = cand
        return best

    best: Optional[AcceptHit] = None
    best = record_accept(i, cur, best)

    while i < len(text):
        ch = text[i]
        before = fmt_states(cur)
        nxt = big.move(cur, ch)
        if not nxt:
            break
        cur = big.eps_closure(nxt)
        after = fmt_states(cur)
        trace.append(StepTrace(i, repr(ch), before, after))
        i += 1
        best = record_accept(i, cur, best)

    return best, trace


# =========================================
# предобработка: удаление //комментариев (игнорируем их)
# =========================================

def strip_line_comments(src: str) -> str:
    out_lines: List[str] = []
    for line in src.splitlines():
        if "//" in line:
            line = line.split("//", 1)[0]
        out_lines.append(line)
    return "\n".join(out_lines)


# =========================================
# токенизация входа с трассировками по токенам
# =========================================

def tokenize_with_traces(text: str, big: ENFA) -> Tuple[List[Token], List[str], List[str], List[Tuple[Token, List[StepTrace]]]]:
    """
    возвращает:
      - список токенов,
      - таблицу идентификаторов (без повторов),
      - таблицу констант (без повторов),
      - список (токен, трассировка его чтения ε-nfa).
    """
    text = strip_line_comments(text)

    tokens: List[Token] = []
    idents: List[str] = []
    consts: List[str] = []
    token_traces: List[Tuple[Token, List[StepTrace]]] = []

    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue

        hit, trace = run_maximal_munch_with_trace(big, text, i)
        if hit is None or len(hit.lexeme) == 0:
            print(f"предупреждение: нераспознанный символ '{text[i]}' на позиции {i}")
            i += 1
            continue

        # keyword vs identifier и пост-валидации
        if hit.category == TokenCat.IDENT and hit.lexeme in KEYWORDS:
            tok = Token(TokenCat.KEYWORD, hit.lexeme)
        elif hit.category == TokenCat.IDENT:
            if len(hit.lexeme) > IDENT_MAXLEN:
                print(f"предупреждение: идентификатор '{hit.lexeme}' длиннее {IDENT_MAXLEN} символов")
            tok = Token(TokenCat.IDENT, hit.lexeme)
            if hit.lexeme not in idents:
                idents.append(hit.lexeme)
        elif hit.category == TokenCat.NUMBER:
            tok = Token(TokenCat.NUMBER, hit.lexeme)
            try:
                val = int(hit.lexeme)
                if not (INT_MIN <= val <= INT_MAX):
                    print(f"предупреждение: число {hit.lexeme} вне диапазона [{INT_MIN}..{INT_MAX}]")
            except ValueError:
                print(f"предупреждение: некорректная числовая лексема '{hit.lexeme}'")
            if hit.lexeme not in consts:
                consts.append(hit.lexeme)
        else:
            tok = Token(hit.category, hit.lexeme)

        tokens.append(tok)
        token_traces.append((tok, trace))
        i = hit.end_index

    return tokens, idents, consts, token_traces


# =========================================
# печать результатов
# =========================================

def print_tokens(tokens: List[Token]):
    rows = [(t.category.value, t.value) for t in tokens]
    print("\nлексемы:")
    print(tabulate(rows or [["—","—"]], headers=["category", "value"], tablefmt="github",
                   colalign=("center","left")))

def print_tables(identifiers: List[str], numbers: List[str]):
    print("\nтаблица идентификаторов:")
    print(tabulate([[s] for s in identifiers] or [["—"]], headers=["identifier"], tablefmt="github"))
    print("\nтаблица констант:")
    print(tabulate([[s] for s in numbers] or [["—"]], headers=["integer"], tablefmt="github"))

def print_token_traces(token_traces: List[Tuple[Token, List[StepTrace]]]):
    print("\nпошаговая трассировка ε-nfa для каждого токена:")
    if not token_traces:
        print("  (нет токенов — вход пуст или состоит из пробелов/комментариев)")
        return
    for i, (tok, tr) in enumerate(token_traces, 1):
        print(f"\n[{i}] {tok.category.value}: '{tok.value}'")
        if not tr:
            print("  (ε-принятие без чтения символов)")
            continue
        rows = [[st.pos, st.char, st.before, st.after] for st in tr]
        print(tabulate(rows, headers=["pos", "char", "states_before", "states_after"], tablefmt="github",
                       colalign=("right","center","left","left")))


# =========================================
# визуализация объединённого автомата (dot + png)
# =========================================

def draw_automaton(big: ENFA, filename: str = "enfa_full"):
    """
    сохраняет .dot всегда; пытается построить .png, если установлен python-graphviz и system graphviz.
    """
    # формируем .dot вручную (полезно для отчёта/диагностики)
    dot_lines: List[str] = ["// ε-nfa combined", "digraph G {", "  rankdir=LR;"]
    # узлы
    for q in sorted(big.states):
        label = f"q{q}"
        if big.start == q:
            label = "→" + label
        shape = "doublecircle" if q in big.accept_info else "circle"
        dot_lines.append(f'  q{q} [label="{label}" shape={shape}];')
    # рёбра (точные)
    for s in sorted(big.states):
        for lab, lst in big.delta.get(s, {}).items():
            for t in lst:
                lab_safe = lab.replace('"', '\\"')
                dot_lines.append(f'  q{s} -> q{t} [label="{lab_safe}"];')
        # рёбра (предикаты)
        for (_pred, name, t) in big.pred_delta.get(s, []):
            dot_lines.append(f'  q{s} -> q{t} [label="<{name}>"];')
    dot_lines.append("}")

    with open(f"{filename}.dot", "w", encoding="utf-8") as f:
        f.write("\n".join(dot_lines))
    print(f"\n[ok] dot-файл сохранён: {filename}.dot")

    # попробуем сразу сгенерировать png
    if not HAS_GRAPHVIZ:
        print("[i] png не генерируется: пакет 'graphviz' не доступен в python. установите 'pip install graphviz' и system graphviz.")
        return
    try:
        g = Digraph(comment="ε-nfa full", format="png")
        g.attr(rankdir="LR")
        for q in sorted(big.states):
            shape = "doublecircle" if q in big.accept_info else "circle"
            label = f"q{q}"
            if big.start == q:
                label = "→" + label
            g.node(f"q{q}", label=label, shape=shape)
        for s in sorted(big.states):
            for lab, lst in big.delta.get(s, {}).items():
                for t in lst:
                    g.edge(f"q{s}", f"q{t}", label=lab)
            for (_pred, name, t) in big.pred_delta.get(s, []):
                g.edge(f"q{s}", f"q{t}", label=f"<{name}>")
        outpath = g.render(filename=filename, cleanup=True)
        print(f"[ok] png с графом автомата: {outpath}")
    except Exception as e:
        print(f"[i] не удалось сгенерировать png через graphviz: {e}")
        print("    вы можете сгенерировать вручную: dot -Tpng enfa_full.dot -o enfa_full.png")


# =========================================
# формирование проекта: под-автоматы + общий ε-nfa
# =========================================

def build_all_submachines() -> Dict[str, ENFA]:
    """
    строит и возвращает словарь под-автоматов для наглядной печати δ-таблиц.
    """
    subs: Dict[str, ENFA] = {
        "KW_do":     build_keyword_nfa("do"),
        "KW_while":  build_keyword_nfa("while"),
        "KW_loop":   build_keyword_nfa("loop"),
        "IDENT":     build_identifier_nfa(),
        "NUMBER":    build_number_nfa(),
        "OPERATORS": build_operator_nfa_list(),
        "SEPARATORS":build_separator_nfa_list(),
    }
    return subs

def build_full_automaton_from_subs(subs: Dict[str, ENFA]) -> ENFA:
    """
    объединяет все под-автоматы в большой ε-nfa.
    """
    order = ["KW_do", "KW_while", "KW_loop", "IDENT", "NUMBER", "OPERATORS", "SEPARATORS"]
    return union_enfas([subs[name] for name in order])


def save_report(subs, big, tokens, idents, consts, token_traces, filename="report.md"):
    with open(filename, "w", encoding="utf-8") as f:
        def w(s=""): f.write(s + "\n")

        # δ-таблицы под-автоматов
        for name, e in subs.items():
            w(f"## δ-таблица автомата {name}\n")
            labels = []
            # соберём таблицу строками так же как print_delta_table, но в Markdown
            from io import StringIO
            buf = StringIO()
            e.print_delta_table("", show_predicates=True)
            # у нас print_delta_table пишет сразу в stdout,
            # поэтому для красоты лучше переписать аналог в отдельную функцию,
            # но пока можно вызвать и перехватить — оставим как упражнение
            # или временно не включать
            w()

        # δ-таблица объединённого автомата
        w("## δ-таблица объединённого ε-NFA\n")
        # проще всего: вызвать big.print_delta_table и перенаправить stdout в файл
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            big.print_delta_table("таблица", show_predicates=True)
        w("```")
        w(buf.getvalue().strip())
        w("```")

        # лексемы
        w("\n## Лексемы\n")
        w(tabulate([(t.category.value, t.value) for t in tokens],
                   headers=["category","value"], tablefmt="github"))

        # таблицы идентификаторов и констант
        w("\n## Таблица идентификаторов\n")
        w(tabulate([[s] for s in idents], headers=["identifier"], tablefmt="github"))
        w("\n## Таблица констант\n")
        w(tabulate([[s] for s in consts], headers=["integer"], tablefmt="github"))

        # трассировки
        w("\n## Пошаговые трассировки по токенам\n")
        for i,(tok,tr) in enumerate(token_traces,1):
            w(f"### [{i}] {tok.category.value}: '{tok.value}'\n")
            if not tr:
                w("(ε-принятие без чтения символов)\n")
            else:
                rows = [[st.pos, st.char, st.before, st.after] for st in tr]
                w(tabulate(rows, headers=["pos","char","states_before","states_after"], tablefmt="github"))
                w()

    print(f"[ok] отчет сохранен в {filename}")


# =========================================
# main: демонстрация на примере (как в задании)
# =========================================

if __name__ == "__main__":
    # исходная программа (пример из задания; добавлены комментарии для демонстрации игнора)
    program = """\
do while x < 10    // цикл увеличения
    x = x + 1;
loop
"""

    # 1) строим под-автоматы и печатаем их δ-таблицы (это важно для отчёта)
    subs = build_all_submachines()
    subs["KW_do"].print_delta_table("δ-таблица: ключевое слово 'do'")
    subs["KW_while"].print_delta_table("δ-таблица: ключевое слово 'while'")
    subs["KW_loop"].print_delta_table("δ-таблица: ключевое слово 'loop'")
    subs["IDENT"].print_delta_table("δ-таблица: идентификатор (letter (letter|digit)*)")
    subs["NUMBER"].print_delta_table("δ-таблица: число ([+-]? digit+)")
    subs["OPERATORS"].print_delta_table("δ-таблица: операторы (=,==,<,>,<=,>=,<>,+,-,*,/)")
    subs["SEPARATORS"].print_delta_table("δ-таблица: разделители (;,(,))")

    # 2) объединяем в один ε-nfa и печатаем объединённую δ-таблицу
    big = build_full_automaton_from_subs(subs)
    big.print_delta_table("δ-таблица: объединённый ε-nfa", show_predicates=True)

    # 3) токенизация с трассировками
    tokens, idents, consts, token_traces = tokenize_with_traces(program, big)

    # 4) печать результатов (лексемы и таблицы)
    print_tokens(tokens)
    print_tables(idents, consts)

    # 5) печать пошаговой трассировки по каждому токену
    print_token_traces(token_traces)

    # 6) сохраняем .dot и (если возможно) .png с графом объединённого автомата
    draw_automaton(big, filename="enfa_full")

    save_report(subs, big, tokens, idents, consts, token_traces, filename="report.md")

