import enum
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Callable
from tabulate import tabulate

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except Exception:
    HAS_GRAPHVIZ = False


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


KEYWORDS = {"do", "while", "loop"}

CATEGORY_PRIORITY = {
    TokenCat.KEYWORD:   5,
    TokenCat.IDENT:     4,
    TokenCat.NUMBER:    3,
    TokenCat.OPERATOR:  2,
    TokenCat.SEPARATOR: 1,
}

IDENT_MAXLEN = 255
INT_MIN, INT_MAX = -32768, 32767


class ENFA:
    def __init__(self):
        self.next_state_id: int = 0
        self.start: Optional[int] = None
        self.states: Set[int] = set()
        self.delta: Dict[int, Dict[str, List[int]]] = {}
        self.pred_delta: Dict[int, List[Tuple[Callable[[str], bool], str, int]]] = {}
        self.accept_info: Dict[int, Tuple[TokenCat, Optional[str]]] = {}

    def _new_state(self) -> int:
        s = self.next_state_id
        self.next_state_id += 1
        self.states.add(s)
        return s

    def _add_edge(self, s_from: int, label: str, s_to: int):
        self.delta.setdefault(s_from, {}).setdefault(label, []).append(s_to)

    def _add_pred_edge(self, s_from: int, pred: Callable[[str], bool], pred_name: str, s_to: int):
        self.pred_delta.setdefault(s_from, []).append((pred, pred_name, s_to))

    def add_string_nfa(self, s: str, category: TokenCat, tag: Optional[str] = None) -> Tuple[int, int]:
        cur = self._new_state()
        start = cur
        for ch in s:
            nxt = self._new_state()
            self._add_edge(cur, ch, nxt)
            cur = nxt
        self.accept_info[cur] = (category, tag if tag is not None else s)
        return start, cur

    def mark_accept(self, state: int, category: TokenCat, tag: Optional[str] = None):
        self.accept_info[state] = (category, tag)

    def eps_closure(self, S: Set[int]) -> Set[int]:
        st = set(S)
        stack = list(S)
        while stack:
            q = stack.pop()
            for nxt in self.delta.get(q, {}).get("ε", []):
                if nxt not in st:
                    st.add(nxt)
                    stack.append(nxt)
        return st

    def move(self, S: Set[int], ch: str) -> Set[int]:
        out = set()
        for q in S:
            for nxt in self.delta.get(q, {}).get(ch, []):
                out.add(nxt)
            for pred, _name, nxt in self.pred_delta.get(q, []):
                if pred(ch):
                    out.add(nxt)
        return out


def is_letter(ch: str) -> bool:
    return ch.isalpha()

def is_digit(ch: str) -> bool:
    return ch.isdigit()

def is_sign(ch: str) -> bool:
    return ch in "+-"


def build_keyword_nfa(word: str) -> ENFA:
    e = ENFA()
    s, _ = e.add_string_nfa(word, TokenCat.KEYWORD, tag=word)
    e.start = s
    return e

def build_identifier_nfa() -> ENFA:
    e = ENFA()
    q0 = e._new_state(); e.start = q0
    q1 = e._new_state()
    e._add_pred_edge(q0, is_letter, "LETTER", q1)
    e._add_pred_edge(q1, is_letter, "LETTER", q1)
    e._add_pred_edge(q1, is_digit,  "DIGIT",  q1)
    e.mark_accept(q1, TokenCat.IDENT, "ID")
    return e

def build_number_nfa() -> ENFA:
    e = ENFA()
    q0 = e._new_state(); e.start = q0
    q_sign = e._new_state()
    e._add_pred_edge(q0, is_sign, "SIGN", q_sign)
    q1 = e._new_state()
    e._add_pred_edge(q0,     is_digit, "DIGIT", q1)
    e._add_pred_edge(q_sign, is_digit, "DIGIT", q1)
    e._add_pred_edge(q1, is_digit, "DIGIT", q1)
    e.mark_accept(q1, TokenCat.NUMBER, "INT")
    return e

def build_operator_nfa_list() -> ENFA:
    ops = ["==", "<=", ">=", "<>", "=", "<", ">", "+", "-", "*", "/"]
    e = ENFA()
    starts: List[Tuple[int, int]] = []
    for op in ops:
        s, _ = e.add_string_nfa(op, TokenCat.OPERATOR, tag=op)
        starts.append((s, _))
    s0 = e._new_state(); e.start = s0
    for s, _a in starts:
        e._add_edge(s0, "ε", s)
    return e

def build_separator_nfa_list() -> ENFA:
    seps = [";", "(", ")"]
    e = ENFA()
    s0 = e._new_state(); e.start = s0
    for sp in seps:
        s, _ = e.add_string_nfa(sp, TokenCat.SEPARATOR, tag=sp)
        e._add_edge(s0, "ε", s)
    return e


def union_enfas(enfas: List[ENFA]) -> ENFA:
    big = ENFA()
    def copy_enfa(src: ENFA) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for s in sorted(src.states):
            ns = big._new_state()
            mapping[s] = ns
        for s in src.states:
            for lab, lst in src.delta.get(s, {}).items():
                for t in lst:
                    big._add_edge(mapping[s], lab, mapping[t])
            for (pred, name, t) in src.pred_delta.get(s, []):
                big._add_pred_edge(mapping[s], pred, name, mapping[t])
        for s, info in src.accept_info.items():
            big.accept_info[mapping[s]] = info
        return mapping
    maps = [copy_enfa(e) for e in enfas]
    s0 = big._new_state(); big.start = s0
    for e, m in zip(enfas, maps):
        big._add_edge(s0, "ε", m[e.start])
    return big


@dataclass
class StepTrace:
    pos: int
    char: str
    before: str
    after: str

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


def strip_line_comments(src: str) -> str:
    out_lines: List[str] = []
    for line in src.splitlines():
        if "//" in line:
            line = line.split("//", 1)[0]
        out_lines.append(line)
    return "\n".join(out_lines)


def tokenize_with_traces(text: str, big: ENFA) -> Tuple[List[Token], List[str], List[str], List[Tuple[Token, List[StepTrace]]]]:
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


def draw_automaton(big: ENFA, filename: str = "enfa_full"):
    dot_lines: List[str] = ["// ε-nfa combined", "digraph G {", "  rankdir=LR;"]
    for q in sorted(big.states):
        label = f"q{q}"
        if big.start == q:
            label = "→" + label
        shape = "doublecircle" if q in big.accept_info else "circle"
        dot_lines.append(f'  q{q} [label="{label}" shape={shape}];')
    for s in sorted(big.states):
        for lab, lst in big.delta.get(s, {}).items():
            for t in lst:
                lab_safe = lab.replace('"', '\\"')
                dot_lines.append(f'  q{s} -> q{t} [label="{lab_safe}"];')
        for (_pred, name, t) in big.pred_delta.get(s, []):
            dot_lines.append(f'  q{s} -> q{t} [label="<{name}>"];')
    dot_lines.append("}")
    with open(f"{filename}.dot", "w", encoding="utf-8") as f:
        f.write("\n".join(dot_lines))
    print(f"\n[ok] dot-файл сохранён: {filename}.dot")
    if not HAS_GRAPHVIZ:
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
        g.render(filename=filename, cleanup=True)
    except Exception:
        pass


def build_all_submachines() -> Dict[str, ENFA]:
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
    order = ["KW_do", "KW_while", "KW_loop", "IDENT", "NUMBER", "OPERATORS", "SEPARATORS"]
    return union_enfas([subs[name] for name in order])


if __name__ == "__main__":
    program = """\
do while x < 10
    x = x + 1;
loop
"""
    subs = build_all_submachines()
    big = build_full_automaton_from_subs(subs)
    draw_automaton(big, filename="enfa_full")
