/* logic_engine.hpp - a small, dependency-free Prolog/Datalog-style reasoner.

   WHY THIS EXISTS
   ---------------
   The deliberation stack is probabilistic end to end: the 8B model writes
   prose, the GNN supplies associations, the tiny expert scores text.  None of
   them can answer a question whose answer must be *certain* - "is this plan
   consistent with the stated constraints?", "which boxes are still open?",
   "does goal X imply prerequisite Y?".  Phoenix therefore lacked a symbolic
   layer that can prove or refute a claim.

   This engine is that layer.  It is deliberately tiny and dependency-free
   (no SWI-Prolog install, no MCP server, no network):
     * facts + Horn-clause rules over first-order terms;
     * forward chaining (semi-naive style: re-evaluate rules until the fact
       set reaches a fixpoint) for Datalog-style closure queries;
     * backward chaining (SLD resolution with a depth bound) for "is there
       an X such that p(X)?" questions;
     * builtin comparison / equality literals evaluated on ground terms;
     * a JSON façade (runJson) so the gateway and the AGI action registry can
       call it as a tool without a parser of their own.

   Deliberate limits (documented, not silent):
     - no negation-as-failure, no cut, no arithmetic expressions (a factual
       rule base does not need them and NAF silently changes semantics);
     - comparisons require ground numeric atoms;
     - every derivation is bounded (facts / iterations / solutions / depth),
       so a cyclic rule base fails closed instead of hanging a mission tick.

   Literature this follows (see doc/v10.1/logic_engine.md):
     - Euclid-MCP (deterministic logical reasoning exposed as a tool),
     - NPC decision making with Prolog + ontologies (HAL 2024),
     - the classic Datalog semi-naive evaluation result (fixpoint = closure).
*/
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace logic {

using json = nlohmann::json;

/* ------------------------------------------------------------------ terms */

struct Term {
    enum class Kind { Var, Atom, Num };
    Kind kind{Kind::Atom};
    std::string name; /* variable name (no leading '?') or atom text */
    double num{0.0};

    static Term var(std::string n) {
        Term t;
        t.kind = Kind::Var;
        t.name = std::move(n);
        return t;
    }
    static Term atom(std::string a) {
        Term t;
        t.kind = Kind::Atom;
        t.name = std::move(a);
        return t;
    }
    static Term number(double v) {
        Term t;
        t.kind = Kind::Num;
        t.num = v;
        return t;
    }
    bool isVar() const { return kind == Kind::Var; }
    bool isNum() const { return kind == Kind::Num; }
    /* Ground identity used for fact de-duplication / output keys. */
    std::string key() const {
        if (kind == Kind::Var) return "?" + name;
        if (kind == Kind::Num) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.17g", num);
            return std::string(buf);
        }
        return name;
    }
    bool operator==(const Term &o) const {
        if (kind != o.kind) return false;
        if (kind == Kind::Num) return num == o.num;
        return name == o.name;
    }
};

struct Literal {
    std::string pred;
    std::vector<Term> args;
};

struct Rule {
    Literal head;
    std::vector<Literal> body;
};

using Binding = std::map<std::string, Term>;

inline void bindApply(const Binding &b, Literal &lit) {
    for (auto &a : lit.args) {
        if (!a.isVar()) continue;
        auto it = b.find(a.name);
        if (it != b.end()) a = it->second;
    }
}

inline bool isBuiltin(const std::string &p) {
    return p == "=" || p == "!=" || p == ">" || p == "<" || p == ">=" ||
           p == "=<" || p == "true";
}

/* Unify two terms, extending the binding.
   ONE-DIRECTIONAL: only one side is ever bound.  Binding both ways
   (bind[X]=Y and bind[Y]=X) makes the deref loop below cycle forever - it kept
   dereferencing X -> Y -> X.  That was the hang the logic tests hit (a gtest
   run burned minutes of CPU before being killed). */
inline bool unify(const Term &a, const Term &b, Binding &bind) {
    Term x = a, y = b;
    /* deref with a hard step cap so a corrupted binding cannot spin */
    for (int guard = 0; guard < 64; ++guard) {
        bool moved = false;
        if (x.isVar()) {
            auto it = bind.find(x.name);
            if (it != bind.end()) {
                x = it->second;
                moved = true;
            }
        }
        if (y.isVar()) {
            auto it = bind.find(y.name);
            if (it != bind.end()) {
                y = it->second;
                moved = true;
            }
        }
        if (!moved) break;
    }
    if (x.isVar() && y.isVar() && x.name == y.name) return true;
    if (!x.isVar() && !y.isVar()) return x == y;
    if (x.isVar()) {
        bind[x.name] = y;
        return true;
    }
    bind[y.name] = x;
    return true;
}

/* Match a (possibly non-ground) literal pattern against a ground fact. */
inline bool matchLiteral(const Literal &pat, const Literal &fact, Binding &bind) {
    if (pat.pred != fact.pred) return false;
    if (pat.args.size() != fact.args.size()) return false;
    for (size_t i = 0; i < pat.args.size(); ++i)
        if (!unify(pat.args[i], fact.args[i], bind)) return false;
    return true;
}

/* Evaluate a builtin literal under a binding.  Non-ground comparisons fail. */
inline bool evalBuiltin(const Literal &lit, const Binding &in, Binding &out) {
    out = in;
    if (lit.pred == "true") return true;
    if (lit.args.size() != 2) return false;
    Term a = lit.args[0], b = lit.args[1];
    if (a.isVar()) {
        auto it = out.find(a.name);
        if (it == out.end()) return false;
        a = it->second;
    }
    if (b.isVar()) {
        auto it = out.find(b.name);
        if (it == out.end()) return false;
        b = it->second;
    }
    if (lit.pred == "=") {
        if (a.isVar() || b.isVar()) return unify(a, b, out);
        return a == b;
    }
    if (lit.pred == "!=") {
        if (a.isVar() || b.isVar()) return false;
        return !(a == b);
    }
    /* numeric comparisons: both sides must resolve to numbers */
    double x = 0.0, y = 0.0;
    bool xn = a.isNum(), yn = b.isNum();
    if (xn) x = a.num;
    else {
        /* allow numeric atoms ("3") so text-extracted facts work */
        try {
            size_t used = 0;
            x = std::stod(a.name, &used);
            xn = used == a.name.size();
        } catch (...) {
        }
    }
    if (yn) y = b.num;
    else {
        try {
            size_t used = 0;
            y = std::stod(b.name, &used);
            yn = used == b.name.size();
        } catch (...) {
        }
    }
    if (!xn || !yn) return false;
    if (lit.pred == ">") return x > y;
    if (lit.pred == "<") return x < y;
    if (lit.pred == ">=") return x >= y;
    if (lit.pred == "=<") return x <= y;
    return false;
}

/* ----------------------------------------------------------------- engine */

class Engine {
public:
    void addFact(const Literal &l) {
        const std::string k = factKey(l);
        if (factKeys_.insert(k).second) facts_.push_back(l);
    }
    void addRule(const Rule &r) { rules_.push_back(r); }

    size_t factCount() const { return facts_.size(); }
    const std::vector<Literal> &facts() const { return facts_; }

    /* Forward chaining to a fixpoint.  Returns the number of new facts. */
    size_t saturate(size_t maxIterations = 200, size_t maxFacts = 20000,
                    size_t *iterationsUsed = nullptr) {
        size_t produced = 0;
        size_t used = 0;
        for (size_t iter = 0; iter < maxIterations; ++iter) {
            used = iter + 1;
            bool grew = false;
            for (const auto &rule : rules_) {
                std::vector<Binding> envs;
                bindSolve(rule.body, 0, Binding{}, envs, 512);
                for (const auto &env : envs) {
                    Literal head = rule.head;
                    bindApply(env, head);
                    if (!isGround(head)) continue;
                    const std::string k = factKey(head);
                    if (!factKeys_.insert(k).second) continue;
                    facts_.push_back(head);
                    ++produced;
                    grew = true;
                    if (facts_.size() >= maxFacts) {
                        if (iterationsUsed) *iterationsUsed = used;
                        return produced;
                    }
                }
            }
            if (!grew) break;
        }
        if (iterationsUsed) *iterationsUsed = used;
        return produced;
    }

    /* Backward chaining (SLD).  Collects up to maxSolutions bindings.
       maxSteps bounds the TOTAL number of rule applications: a depth bound
       alone is not enough - a cyclic KB (p:-q, q:-p) branches into every
       rule at every level, so depth 24 is 2^24 resolutions and the calling
       thread looks hung (observed: a gtest run burned minutes of CPU).
       Fail closed: once the budget is gone the query returns what it has. */
    std::vector<Binding> solve(const Literal &goal, size_t maxSolutions = 16,
                               int maxDepth = 24, size_t maxSteps = 20000) {
        std::vector<Binding> out;
        std::vector<Literal> goals{goal};
        size_t steps = maxSteps;
        sld(goals, 0, Binding{}, maxDepth, maxSolutions, steps, out);
        return out;
    }

    static bool isGround(const Literal &l) {
        for (const auto &a : l.args)
            if (a.isVar()) return false;
        return true;
    }

private:
    std::vector<Literal> facts_;
    std::set<std::string> factKeys_;
    std::vector<Rule> rules_;

    static std::string factKey(const Literal &l) {
        std::string k = l.pred;
        k.push_back('(');
        for (size_t i = 0; i < l.args.size(); ++i) {
            if (i) k.push_back(',');
            k += l.args[i].key();
        }
        k.push_back(')');
        return k;
    }

    /* Resolve a conjunction of literals against the known facts. */
    void bindSolve(const std::vector<Literal> &body, size_t idx,
                   const Binding &env, std::vector<Binding> &out,
                   size_t cap) const {
        if (out.size() >= cap) return;
        if (idx == body.size()) {
            out.push_back(env);
            return;
        }
        const Literal &lit = body[idx];
        if (isBuiltin(lit.pred)) {
            Binding next;
            if (!evalBuiltin(lit, env, next)) return;
            bindSolve(body, idx + 1, next, out, cap);
            return;
        }
        Literal pat = lit;
        bindApply(env, pat);
        for (const auto &fact : facts_) {
            Binding b = env;
            if (!matchLiteral(pat, fact, b)) continue;
            bindSolve(body, idx + 1, b, out, cap);
            if (out.size() >= cap) return;
        }
    }

    /* SLD resolution: prove goals[0], then the rest. */
    void sld(const std::vector<Literal> &goals, size_t idx, const Binding &env,
             int maxDepth, size_t maxSolutions, size_t &steps,
             std::vector<Binding> &out) const {
        if (out.size() >= maxSolutions) return;
        if (steps == 0) return;
        if (idx >= goals.size()) {
            out.push_back(env);
            return;
        }
        if (maxDepth <= 0) return;
        Literal goal = goals[idx];
        bindApply(env, goal);

        if (isBuiltin(goal.pred)) {
            Binding next;
            if (!evalBuiltin(goal, env, next)) return;
            sld(goals, idx + 1, next, maxDepth - 1, maxSolutions, steps, out);
            return;
        }
        /* facts */
        if (isGround(goal)) {
            if (factKeys_.count(factKey(goal)))
                sld(goals, idx + 1, env, maxDepth - 1, maxSolutions, steps, out);
        } else {
            for (const auto &fact : facts_) {
                Binding b = env;
                if (!matchLiteral(goal, fact, b)) continue;
                sld(goals, idx + 1, b, maxDepth - 1, maxSolutions, steps, out);
                if (out.size() >= maxSolutions || steps == 0) return;
            }
        }
        /* rules whose head may match */
        for (const auto &rule : rules_) {
            if (rule.head.pred != goal.pred) continue;
            if (rule.head.args.size() != goal.args.size()) continue;
            Binding b = env;
            /* Rename the rule apart so each use is independent, but head and
               body MUST share one rename map: renaming them separately gave
               the head X#1 and the body X#2, so the head's bindings never
               reached the body and every rule silently produced zero
               solutions (this was the Sample2/3/7 failure). */
            std::map<std::string, std::string> ren;
            auto renameTerm = [&](const Term &t) -> Term {
                if (!t.isVar()) return t;
                auto it = ren.find(t.name);
                if (it == ren.end())
                    it = ren
                             .emplace(t.name, t.name + "#" +
                                                  std::to_string(nextRename()))
                             .first;
                return Term::var(it->second);
            };
            auto renameLit = [&](const Literal &l) {
                Literal o;
                o.pred = l.pred;
                o.args.reserve(l.args.size());
                for (const auto &a : l.args) o.args.push_back(renameTerm(a));
                return o;
            };
            const Literal head = renameLit(rule.head);
            if (!matchLiteral(goal, head, b)) continue;
            if (steps == 0) return;
            --steps;
            std::vector<Literal> body;
            body.reserve(rule.body.size());
            for (const auto &l : rule.body) body.push_back(renameLit(l));
            std::vector<Literal> next;
            next.reserve(body.size() + goals.size() - idx - 1);
            next.insert(next.end(), body.begin(), body.end());
            next.insert(next.end(), goals.begin() + static_cast<long>(idx) + 1,
                        goals.end());
            sld(next, 0, b, maxDepth - 1, maxSolutions, steps, out);
            if (out.size() >= maxSolutions) return;
        }
    }

    /* Rename counter: thread_local, so two mission threads resolving rules
       never share (or race on) fresh-variable names. */
    static int nextRename() {
        static thread_local int n = 0;
        return ++n;
    }
};

/* -------------------------------------------------------------- JSON FAÇADE */

namespace detail {

inline bool isPlainNumber(const std::string &s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i = 1;
    if (i >= s.size()) return false;
    bool dot = false, digit = false;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (c >= '0' && c <= '9') {
            digit = true;
        } else if (c == '.' && !dot) {
            dot = true;
        } else {
            return false;
        }
    }
    return digit;
}

inline std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

/* "parent(tom, bob)" / "ancestor(X,?Y)" / "age(alice, 34)". */
inline bool parseLiteralText(const std::string &text, Literal &out,
                            std::string &err) {
    const std::string s = trim(text);
    const auto lp = s.find('(');
    if (lp == std::string::npos || s.empty() || s.back() != ')') {
        err = "literal must look like pred(a,b): " + s;
        return false;
    }
    out.pred = trim(s.substr(0, lp));
    const bool namedOk =
        !out.pred.empty() &&
        (std::isalpha(static_cast<unsigned char>(out.pred[0])) || out.pred[0] == '_');
    if (!namedOk && !isBuiltin(out.pred)) {
        err = "bad predicate name in: " + s;
        return false;
    }
    std::string args = s.substr(lp + 1, s.size() - lp - 2);
    out.args.clear();
    std::string cur;
    int depth = 0;
    bool quoted = false;
    std::vector<std::string> parts;
    for (char c : args) {
        if (c == '"' || c == '\'') quoted = !quoted;
        if (!quoted && (c == '(' || c == '[')) ++depth;
        if (!quoted && (c == ')' || c == ']')) --depth;
        if (!quoted && depth == 0 && c == ',') {
            parts.push_back(cur);
            cur.clear();
            continue;
        }
        cur.push_back(c);
    }
    if (!trim(cur).empty() || !parts.empty()) parts.push_back(cur);
    for (auto &p : parts) {
        std::string t = trim(p);
        if (t.empty()) {
            err = "empty argument in: " + s;
            return false;
        }
        const bool quotedArg = t.size() >= 2 && ((t.front() == '"' && t.back() == '"') ||
                                                 (t.front() == '\'' && t.back() == '\''));
        if (quotedArg) {
            out.args.push_back(Term::atom(t.substr(1, t.size() - 2)));
        } else if (t.front() == '?' ) {
            out.args.push_back(Term::var(trim(t.substr(1))));
        } else if (std::isupper(static_cast<unsigned char>(t.front())) ||
                   t.front() == '_') {
            out.args.push_back(Term::var(t));
        } else if (isPlainNumber(t)) {
            out.args.push_back(Term::number(std::stod(t)));
        } else {
            out.args.push_back(Term::atom(t));
        }
    }
    return true;
}

inline bool literalFromJson(const json &v, Literal &out, std::string &err) {
    if (v.is_string()) return parseLiteralText(v.get<std::string>(), out, err);
    if (!v.is_object()) {
        err = "literal must be a string or an object";
        return false;
    }
    out.pred = v.value("pred", std::string());
    if (out.pred.empty()) {
        err = "literal object needs pred";
        return false;
    }
    out.args.clear();
    if (v.contains("args") && v["args"].is_array()) {
        for (const auto &a : v["args"]) {
            if (a.is_number()) {
                out.args.push_back(Term::number(a.get<double>()));
            } else if (a.is_string()) {
                const std::string t = a.get<std::string>();
                /* explicit variable marker in the object form */
                if (!t.empty() && (t.front() == '?' || std::isupper(static_cast<unsigned char>(t.front()))))
                    out.args.push_back(t.front() == '?' ? Term::var(t.substr(1)) : Term::var(t));
                else if (isPlainNumber(t))
                    out.args.push_back(Term::number(std::stod(t)));
                else
                    out.args.push_back(Term::atom(t));
            } else if (a.is_object() && a.contains("var")) {
                out.args.push_back(Term::var(a["var"].get<std::string>()));
            } else if (a.is_object() && a.contains("num")) {
                out.args.push_back(Term::number(a["num"].get<double>()));
            } else if (a.is_object() && a.contains("atom")) {
                out.args.push_back(Term::atom(a["atom"].get<std::string>()));
            } else {
                err = "unsupported argument form";
                return false;
            }
        }
    }
    return true;
}

inline json termToJson(const Term &t) {
    if (t.isNum()) return json(t.num);
    return json(t.name);
}

/* Follow a variable chain to its ground value (one-directional unification
   leaves e.g. Who -> X#1 -> alice; printing Who as "X#1" was the S3/S7
   failure).  Returns the raw term when the chain stays unbound. */
inline Term deref(const Binding &b, Term t) {
    for (int guard = 0; guard < 64 && t.isVar(); ++guard) {
        auto it = b.find(t.name);
        if (it == b.end()) break;
        if (it->second.isVar() && it->second.name == t.name) break;
        t = it->second;
    }
    return t;
}

}  // namespace detail

/* JSON entry point used by the gateway tool dispatch.

   Request:
     {
       "facts": ["parent(tom,bob)", {"pred":"age","args":["tom",34]}],
       "rules": [{"head":"ancestor(X,Y)","body":["parent(X,Y)"]},
                 {"head":"ancestor(X,Y)","body":["parent(X,Z)","ancestor(Z,Y)"]}],
       "query": "ancestor(tom,Who)",       // or {"pred":..,"args":[..]}
       "mode":  "backward" | "forward",    // default backward
       "maxSolutions": 8, "maxIterations": 200, "maxDepth": 24
     }
   Response:
     {"ok":true,"mode":"backward","solutions":[{"W":..}],"derived":0,
      "facts":N,"iterations":M}
*/
inline json runJson(const json &payload) {
    json out = json{{"ok", false}};
    if (!payload.is_object()) {
        out["error"] = "payload must be an object";
        return out;
    }
    Engine engine;
    std::string err;
    if (payload.contains("facts") && payload["facts"].is_array()) {
        for (const auto &f : payload["facts"]) {
            Literal lit;
            if (!detail::literalFromJson(f, lit, err)) {
                out["error"] = err;
                return out;
            }
            if (!Engine::isGround(lit)) {
                out["error"] = "facts must be ground (no variables)";
                return out;
            }
            engine.addFact(lit);
        }
    }
    if (payload.contains("rules") && payload["rules"].is_array()) {
        for (const auto &r : payload["rules"]) {
            if (!r.is_object() || !r.contains("head") || !r.contains("body")) {
                out["error"] = "rule needs head and body";
                return out;
            }
            Rule rule;
            if (!detail::literalFromJson(r["head"], rule.head, err)) {
                out["error"] = err;
                return out;
            }
            if (!r["body"].is_array() || r["body"].empty()) {
                out["error"] = "rule body must be a non-empty array";
                return out;
            }
            for (const auto &b : r["body"]) {
                Literal lit;
                if (!detail::literalFromJson(b, lit, err)) {
                    out["error"] = err;
                    return out;
                }
                rule.body.push_back(std::move(lit));
            }
            engine.addRule(rule);
        }
    }
    const std::string mode = payload.value("mode", std::string("backward"));
    const size_t maxSolutions =
        static_cast<size_t>(std::max(1, std::min(64, payload.value("maxSolutions", 8))));
    const size_t maxIterations =
        static_cast<size_t>(std::max(1, std::min(2000, payload.value("maxIterations", 200))));
    const int maxDepth = std::max(1, std::min(64, payload.value("maxDepth", 24)));
    const size_t maxSteps = static_cast<size_t>(
        std::max(64, std::min(500000, payload.value("maxSteps", 20000))));

    size_t derived = 0;
    size_t iterationsUsed = 0;
    if (mode == "forward")
        derived = engine.saturate(maxIterations, 20000, &iterationsUsed);

    json solutions = json::array();
    if (payload.contains("query")) {
        Literal goal;
        if (!detail::literalFromJson(payload["query"], goal, err)) {
            out["error"] = err;
            return out;
        }
        /* forward mode answers a ground query from the closure; a
           non-ground query is still resolved by SLD over the closure. */
        const auto sols = engine.solve(goal, maxSolutions, maxDepth, maxSteps);
        for (const auto &b : sols) {
            /* strip internal rename suffixes (X#12 -> X) and resolve chains */
            json row = json::object();
            for (const auto &kv : b) {
                if (kv.first.empty()) continue;
                const Term ground = detail::deref(b, Term::var(kv.first));
                if (ground.isVar()) continue; /* still unbound: not an answer */
                std::string name = kv.first;
                const auto hash = name.find('#');
                if (hash != std::string::npos) name.resize(hash);
                if (name.empty()) continue;
                row[name] = detail::termToJson(ground);
            }
            solutions.push_back(row);
        }
    }
    out["ok"] = true;
    out["mode"] = mode;
    out["solutions"] = solutions;
    out["solutionCount"] = solutions.size();
    out["derived"] = derived;
    out["factCount"] = engine.factCount();
    out["iterations"] = iterationsUsed;
    return out;
}

}  // namespace logic
}  // namespace phoenix
