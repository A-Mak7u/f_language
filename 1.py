import json
from typing import Dict, List, Set, Tuple

INPUT_FILE = "fa_inputs.json"

from tabulate import tabulate

def print_table(title: str, headers: List[str], rows: List[List[str]]):
    print(f"\n{title}")
    print(tabulate(rows, headers=headers, tablefmt="github"))

def fmt_set(s: Set[str]) -> str:
    return "∅" if not s else "{" + ",".join(sorted(s)) + "}"

def ensure_list(x):
    return x if isinstance(x, list) else [x]

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
        rows = [[q] + [self.delta.get(q, {}).get(a, "-") for a in sorted(self.alphabet)]
                for q in sorted(self.states)]
        return headers, rows

    def run(self, w: str):
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

# NFA (без ε)

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
            row = [q]
            for a in sorted(self.alphabet):
                dest = set(self.delta.get(q, {}).get(a, []))
                row.append(fmt_set(dest) if dest else "-")
            rows.append(row)
        return headers, rows

    def run(self, w: str):
        cur = {self.start}
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
            row = [q]
            for a in sorted(self.alphabet | {"ε"}):
                dest = set(self.delta.get(q, {}).get(a, []))
                row.append(fmt_set(dest) if dest else "-")
            rows.append(row)
        return headers, rows

    def run(self, w: str):
        cur = self.eps_closure({self.start})
        trace: List[Tuple[str, str, str]] = [(fmt_set(cur), "ε", fmt_set(cur))]
        if not cur:
            return False, trace, "ε-замыкание пусто"
        for i, ch in enumerate(w):
            if ch not in self.alphabet:
                trace.append((fmt_set(cur), ch, "-"))
                return False, trace, f"на шаге {i}: символ '{ch}' вне алфавита"
            nxt = self.eps_closure(self.move(cur, ch))
            trace.append((fmt_set(cur), ch, fmt_set(nxt) if nxt else "-"))
            cur = nxt
        trace.append((fmt_set(cur), "ε", fmt_set(cur)))
        ok = bool(cur & self.accepts)
        return ok, trace, ("множество содержит принимающее" if ok
                           else "множество не пересекается с принимающими")

# валидация

def validate_dfa(obj) -> List[str]:
    msgs, states, alpha = [], set(obj["states"]), set(obj["alphabet"])
    start, accepts, delta = obj["start"], set(obj["accepts"]), obj["transitions"]
    if start not in states: msgs.append(f"ошибка: стартовое '{start}' отсутствует")
    if not accepts.issubset(states): msgs.append("ошибка: принимающие вне множества состояний")
    for q, m in delta.items():
        if "ε" in m: msgs.append(f"ошибка: в dfa найден ε-переход в '{q}'")
        for a, t in m.items():
            if a not in alpha: msgs.append(f"ошибка: символ '{a}' вне алфавита")
            if t not in states: msgs.append(f"ошибка: переход '{q}' --{a}--> '{t}' в неизвестное состояние")
    for q in states:
        for a in alpha:
            if a not in delta.get(q, {}):
                msgs.append(f"предупреждение: переход из '{q}' по '{a}' не задан (неполный dfa)")
    return msgs

def validate_nfa(obj) -> List[str]:
    msgs, states, alpha = [], set(obj["states"]), set(obj["alphabet"])
    start, accepts = obj["start"], set(obj["accepts"])
    delta = {q: {sym: ensure_list(d) for sym, d in m.items()} for q, m in obj["transitions"].items()}
    if start not in states: msgs.append(f"ошибка: стартовое '{start}' отсутствует")
    if not accepts.issubset(states): msgs.append("ошибка: принимающие вне множества состояний")
    for q, m in delta.items():
        for a, dests in m.items():
            if a == "ε": msgs.append(f"ошибка: ε-переходы не допускаются в nfa (используйте e-nfa)")
            elif a not in alpha: msgs.append(f"ошибка: символ '{a}' вне алфавита")
            for t in dests:
                if t not in states: msgs.append(f"ошибка: '{q}' --{a}--> '{t}' в неизвестное состояние")
    return msgs

def validate_enfa(obj) -> List[str]:
    msgs, states, alpha = [], set(obj["states"]), set(obj["alphabet"])
    start, accepts = obj["start"], set(obj["accepts"])
    delta = {q: {sym: ensure_list(d) for sym, d in m.items()} for q, m in obj["transitions"].items()}
    if start not in states: msgs.append(f"ошибка: стартовое '{start}' отсутствует")
    if not accepts.issubset(states): msgs.append("ошибка: принимающие вне множества состояний")
    for q, m in delta.items():
        for a, dests in m.items():
            if a != "ε" and a not in alpha: msgs.append(f"ошибка: символ '{a}' вне алфавита")
            for t in dests:
                if t not in states: msgs.append(f"ошибка: '{q}' --{a}--> '{t}' в неизвестное состояние")
    return msgs

# преобразования

def enfa_to_nfa(enfa: ENFA) -> NFA:
    # δ'(q,a) = ε-closure( ⋃_{p ∈ ε-closure(q)} δ(p,a) )
    new_delta: Dict[str, Dict[str, List[str]]] = {}
    new_states = sorted(enfa.states)
    new_alpha  = sorted(enfa.alphabet)
    new_start  = next(iter(enfa.eps_closure({enfa.start})))  # имя стартового не меняем (реально замыкание используется в run)
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
        "start": enfa.start,     # стартовое состояние то же; ε-замыкание учитывается при прогоне
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
            if T_name not in seen:
                seen[T_name] = T
                dfa_states.add(T_name)
                queue.append(T)
    empty_name = "∅"
    if empty_name not in dfa_states:
        dfa_states.add(empty_name)
    dfa_delta.setdefault(empty_name, {})
    for a in sorted(nfa.alphabet):
        dfa_delta[empty_name][a] = empty_name
        for q in list(dfa_states):
            dfa_delta.setdefault(q, {})
            if a not in dfa_delta[q]:
                dfa_delta[q][a] = empty_name
    dfa_obj = {
        "states": sorted(dfa_states),
        "alphabet": sorted(nfa.alphabet),
        "start": start_name,
        "accepts": sorted(dfa_accepts),
        "transitions": dfa_delta
    }
    return DFA(dfa_obj)


def main():
    data = json.load(open(INPUT_FILE, encoding="utf-8"))
    t = data["automaton"]["type"].strip().upper()

    if t in ("DFA",):
        msgs = validate_dfa(data["automaton"])
        A = DFA(data["automaton"])
        derived = []
    elif t in ("NFA",):
        msgs = validate_nfa(data["automaton"])
        A = NFA(data["automaton"])
        dfa_equiv = nfa_to_dfa(A)
        derived = [("эквивалентный dfa (subset construction)", dfa_equiv)]
    elif t in ("E-NFA", "ENFA", "EPSILON-NFA", "EPSILON_NFA"):
        msgs = validate_enfa(data["automaton"])
        A = ENFA(data["automaton"])
        nfa_no_eps = enfa_to_nfa(A)
        dfa_equiv  = nfa_to_dfa(nfa_no_eps)
        derived = [("эквивалентный nfa без ε", nfa_no_eps),
                   ("эквивалентный dfa (после удаления ε)", dfa_equiv)]
    else:
        raise ValueError("тип должен быть DFA, NFA или E-NFA")

    print("\nдиагностика")
    print("\n".join(" - " + m for m in msgs) if msgs else " - ошибок не найдено")

    hdr, rows = A.transition_table()
    print_table("таблица переходов (исходный автомат)", hdr, rows)

    for title, B in derived:
        hdr_b, rows_b = B.transition_table()
        print_table(f"таблица переходов: {title}", hdr_b, rows_b)

    word = input("\nвведите слово: ").strip()

    ok, trace, reason = A.run(word)
    if t == "DFA":
        headers = ["шаг", "текущее", "символ", "следующее"]
        rows = [[str(j), c, s, n] for j, (c, s, n) in enumerate(trace)]
    elif t == "NFA":
        headers = ["шаг", "текущее множество", "символ", "следующее множество"]
        rows = [[str(j), c, s, n] for j, (c, s, n) in enumerate(trace)]
    else:  # E-NFA
        headers = ["шаг", "текущее множество", "символ", "следующее множество"]
        rows = [[str(j), c, s, n] for j, (c, s, n) in enumerate(trace)]

    print(f"\nрезультат (исходный автомат): {'принято' if ok else 'отклонено'} — {reason}")
    print_table("трассировка (исходный автомат)", headers, rows)

    last_verdict = ok
    for title, B in derived:
        ok2, trace2, reason2 = B.run(word)
        if isinstance(B, DFA):
            h2 = ["шаг", "текущее", "символ", "следующее"]
            r2 = [[str(j), c, s, n] for j, (c, s, n) in enumerate(trace2)]
        else:
            h2 = ["шаг", "текущее множество", "символ", "следующее множество"]
            r2 = [[str(j), c, s, n] for j, (c, s, n) in enumerate(trace2)]
        print(f"\nрезультат ({title}): {'принято' if ok2 else 'отклонено'} — {reason2}")
        print_table(f"трассировка ({title})", h2, r2)
        print("сверка: " + ("вердикты совпадают" if last_verdict == ok2 else "вердикты НЕ совпадают"))
        last_verdict = ok2

if __name__ == "__main__":
    main()
