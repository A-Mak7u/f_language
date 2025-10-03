import json
from typing import Dict, List, Set, Tuple
from tabulate import tabulate

INPUT_FILE = "fa_inputs.json"


def print_table(title: str, headers: List[str], rows: List[List[str]]):
    print(f"\n{title}")
    print(tabulate(rows, headers=headers, tablefmt="github", colalign=("center",) * len(headers)))



def fmt_set(s: Set[str]) -> str:
    return "∅" if not s else "{" + ",".join(sorted(s)) + "}"

def ensure_list(x):
    return x if isinstance(x, list) else [x]

def mark_state(q: str, start: str, accepts: Set[str]) -> str:
    prefix = ""
    if q == start:
        prefix += "→"
    if q in accepts:
        prefix += "*"
    return f"{prefix}{q}" if prefix else q

# DFA

class DFA:
    def __init__(self, obj):
        self.states   = set(obj["states"])
        self.alphabet = set(obj["alphabet"])
        self.start    = obj["start"]
        self.accepts  = set(obj["accepts"])
        self.delta: Dict[str, Dict[str, str]] = obj["transitions"]

    def transition_table(self):
        headers = ["состояние"] + sorted(self.alphabet)
        rows = []
        for q in sorted(self.states):
            row = [mark_state(q, self.start, self.accepts)]
            for a in sorted(self.alphabet):
                row.append(self.delta.get(q, {}).get(a, "-"))
            rows.append(row)
        return headers, rows

    def run(self, w: str):
        if w == "":
            ok = self.start in self.accepts
            return ok, [(self.start, "ε", self.start)], \
                   ("пустая строка принята" if ok else "пустая строка отклонена")
        cur = self.start
        trace: List[Tuple[str, str, str]] = []
        for i, ch in enumerate(w):
            if ch not in self.alphabet:
                trace.append((cur, ch, "-"))
                return False, trace, f"на шаге {i}: символ '{ch}' вне алфавита"
            nxt = self.delta.get(cur, {}).get(ch)
            trace.append((cur, ch, nxt if nxt else "-"))
            if nxt is None:
                return False, trace, f"на шаге {i}: переход из '{cur}' по '{ch}' не задан"
            cur = nxt
        trace.append((cur, "ε", cur))
        ok = cur in self.accepts
        return ok, trace, f"закончили в {'принимающем' if ok else 'непринимающем'} состоянии '{cur}'"

# NFA

class NFA:
    def __init__(self, obj):
        self.states   = set(obj["states"])
        self.alphabet = set(obj["alphabet"])
        self.start    = obj["start"]
        self.accepts  = set(obj["accepts"])
        self.delta: Dict[str, Dict[str, List[str]]] = {
            q: {sym: ensure_list(d) for sym, d in m.items()} for q, m in obj["transitions"].items()
        }

    def move(self, S: Set[str], a: str) -> Set[str]:
        out = set()
        for q in S:
            out |= set(self.delta.get(q, {}).get(a, []))
        return out

    def transition_table(self):
        headers = ["состояние"] + sorted(self.alphabet)
        rows = []
        for q in sorted(self.states):
            row = [mark_state(q, self.start, self.accepts)]
            for a in sorted(self.alphabet):
                dest = set(self.delta.get(q, {}).get(a, []))
                row.append(fmt_set(dest) if dest else "-")
            rows.append(row)
        return headers, rows

    def run(self, w: str):
        cur = {self.start}
        if w == "":
            ok = bool(cur & self.accepts)
            return ok, [(fmt_set(cur), "ε", fmt_set(cur))], \
                   ("пустая строка принята" if ok else "пустая строка отклонена")
        trace: List[Tuple[str, str, str]] = [(fmt_set(cur), "ε", fmt_set(cur))]
        for i, ch in enumerate(w):
            if ch not in self.alphabet:
                trace.append((fmt_set(cur), ch, "-"))
                return False, trace, f"на шаге {i}: символ '{ch}' вне алфавита"
            nxt = self.move(cur, ch)
            trace.append((fmt_set(cur), ch, fmt_set(nxt) if nxt else "-"))
            cur = nxt
        trace.append((fmt_set(cur), "ε", fmt_set(cur)))
        ok = bool(cur & self.accepts)
        return ok, trace, ("множество содержит принимающее" if ok
                           else "множество не пересекается с принимающими")

# ε-NFA

class ENFA:
    def __init__(self, obj):
        self.states   = set(obj["states"])
        self.alphabet = set(obj["alphabet"])
        self.start    = obj["start"]
        self.accepts  = set(obj["accepts"])
        self.delta: Dict[str, Dict[str, List[str]]] = {
            q: {sym: ensure_list(d) for sym, d in m.items()} for q, m in obj["transitions"].items()
        }

    def eps_closure(self, S: Set[str]) -> Set[str]:
        st, stack = set(S), list(S)
        while stack:
            q = stack.pop()
            for nxt in self.delta.get(q, {}).get("ε", []):
                if nxt not in st:
                    st.add(nxt); stack.append(nxt)
        return st

    def move(self, S: Set[str], a: str) -> Set[str]:
        out = set()
        for q in S:
            out |= set(self.delta.get(q, {}).get(a, []))
        return out

    def transition_table(self):
        headers = ["состояние"] + sorted(self.alphabet | {"ε"})
        rows = []
        for q in sorted(self.states):
            row = [mark_state(q, self.start, self.accepts)]
            for a in sorted(self.alphabet | {"ε"}):
                dest = set(self.delta.get(q, {}).get(a, []))
                row.append(fmt_set(dest) if dest else "-")
            rows.append(row)
        return headers, rows

    def run(self, w: str):
        cur = self.eps_closure({self.start})  # первый e
        if w == "":
            ok = bool(cur & self.accepts)
            return ok, [(fmt_set(cur), "ε", fmt_set(cur))], \
                   ("пустая строка принята" if ok else "пустая строка отклонена")
        trace: List[Tuple[str, str, str]] = [(fmt_set(cur), "ε", fmt_set(cur))]
        if not cur:
            return False, trace, "ε-замыкание пусто"
        for i, ch in enumerate(w):
            if ch not in self.alphabet:
                trace.append((fmt_set(cur), ch, "-"))
                return False, trace, f"на шаге {i}: символ '{ch}' вне алфавита"
            nxt = self.eps_closure(self.move(cur, ch))  # после каждого символа
            trace.append((fmt_set(cur), ch, fmt_set(nxt) if nxt else "-"))
            cur = nxt
        trace.append((fmt_set(cur), "ε", fmt_set(cur)))  # финальный e
        ok = bool(cur & self.accepts)
        return ok, trace, ("множество содержит принимающее" if ok
                           else "множество не пересекается с принимающими")

# преобразования

def enfa_to_nfa(enfa: ENFA) -> NFA:
    new_delta: Dict[str, Dict[str, List[str]]] = {}
    new_states = sorted(enfa.states)
    new_alpha  = sorted(enfa.alphabet)
    new_accepts: Set[str] = set()
    for q in enfa.states:
        q_cl = enfa.eps_closure({q})
        if q_cl & enfa.accepts:
            new_accepts.add(q)
        new_delta[q] = {}
        for a in enfa.alphabet:
            dest = set()
            for p in q_cl:
                dest |= set(enfa.delta.get(p, {}).get(a, []))
            dest = enfa.eps_closure(dest)
            if dest:
                new_delta[q][a] = sorted(dest)
    nfa_obj = {
        "states": new_states,
        "alphabet": new_alpha,
        "start": enfa.start,
        "accepts": sorted(new_accepts),
        "transitions": new_delta
    }
    return NFA(nfa_obj)

def nfa_to_dfa(nfa: NFA) -> DFA:
    start_set = {nfa.start}
    start_name = fmt_set(start_set)
    queue, seen = [start_set], {start_name: start_set}
    dfa_states, dfa_accepts, dfa_delta = {start_name}, set(), {}

    while queue:
        S = queue.pop(0)
        S_name = fmt_set(S)
        dfa_delta.setdefault(S_name, {})
        if S & nfa.accepts:
            dfa_accepts.add(S_name)
        for a in sorted(nfa.alphabet):
            T = nfa.move(S, a)
            T_name = fmt_set(T)
            dfa_delta[S_name][a] = T_name
            if T and T_name not in seen:   # добавляем только НЕпустые множества
                seen[T_name] = T
                dfa_states.add(T_name)
                queue.append(T)

    return DFA({
        "states": sorted(dfa_states),
        "alphabet": sorted(nfa.alphabet),
        "start": start_name,
        "accepts": sorted(dfa_accepts),
        "transitions": dfa_delta
    })


def main():
    data = json.load(open(INPUT_FILE, encoding="utf-8"))
    t = data["automaton"]["type"].strip().upper()

    if t == "DFA":
        A = DFA(data["automaton"])
        derived = []
    elif t == "NFA":
        A = NFA(data["automaton"])
        dfa_equiv = nfa_to_dfa(A)
        derived = [("эквивалентный dfa", dfa_equiv)]
    elif t in ("E-NFA","ENFA","EPSILON-NFA","EPSILON_NFA"):
        A = ENFA(data["automaton"])
        nfa_no_eps = enfa_to_nfa(A)
        dfa_equiv  = nfa_to_dfa(nfa_no_eps)
        derived = [("эквивалентный nfa без ε", nfa_no_eps),
                   ("эквивалентный dfa", dfa_equiv)]
    else:
        raise ValueError("тип должен быть DFA, NFA или E-NFA")

    hdr, rows = A.transition_table()
    print_table("таблица переходов (исходный автомат)", hdr, rows)

    for title, B in derived:
        hdr_b, rows_b = B.transition_table()
        print_table(f"таблица переходов: {title}", hdr_b, rows_b)

    word = input("\nвведите слово: ").strip()

    ok, trace, reason = A.run(word)
    headers = ["шаг", "текущее", "символ", "следующее"]
    rows = [[str(j), c, s, n] for j, (c, s, n) in enumerate(trace)]
    print(f"\nрезультат (исходный автомат): {'принято' if ok else 'отклонено'} — {reason}")
    print_table("трассировка (исходный автомат)", headers, rows)

    last_verdict = ok
    for title, B in derived:
        ok2, trace2, reason2 = B.run(word)
        headers2 = ["шаг", "текущее", "символ", "следующее"]
        rows2 = [[str(j), c, s, n] for j, (c, s, n) in enumerate(trace2)]
        print(f"\nрезультат ({title}): {'принято' if ok2 else 'отклонено'} — {reason2}")
        print_table(f"трассировка ({title})", headers2, rows2)
        print("сверка: " + ("вердикты совпадают ✅" if last_verdict == ok2 else "вердикты НЕ совпадают ❌"))
        last_verdict = ok2

if __name__ == "__main__":
    main()
