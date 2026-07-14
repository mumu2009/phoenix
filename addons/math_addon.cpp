/* math_addon.cpp - Math addon implementation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

#include "math_addon.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <cstdio>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace addon::builtins {

namespace {

struct Token {
	enum class Type { Number, Op, Func, LParen, RParen, Comma } type{Type::Number};
	std::string text;
	double value{0.0};
	int precedence{0};
	bool rightAssoc{false};
	int arity{0};
};

static bool isIdentChar(char c) {
	return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static int funcArity(const std::string &name) {
	if (name == "min" || name == "max" || name == "pow") return 2;
	return 1;
}

static bool isFuncName(const std::string &name) {
	static const std::unordered_set<std::string> funcs = {
		"sin", "cos", "tan", "tanh", "exp", "log", "sqrt", "abs", "min", "max", "pow"
	};
	return funcs.count(name) > 0;
}

static std::vector<Token> tokenizeExpr(const std::string &expr, bool &ok) {
	ok = true;
	std::vector<Token> out;
	Token prev;
	bool hasPrev = false;
	for (size_t i = 0; i < expr.size();) {
		char c = expr[i];
		if (std::isspace(static_cast<unsigned char>(c))) { i++; continue; }
		if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
			const char *s = expr.c_str() + i;
			char *end = nullptr;
			double v = std::strtod(s, &end);
			if (end == s) { ok = false; return {}; }
			Token t; t.type = Token::Type::Number; t.value = v; t.text = std::string(s, (size_t)(end - s));
			out.push_back(t);
			i = static_cast<size_t>(end - expr.c_str());
			hasPrev = true; prev = t;
			continue;
		}
		if (isIdentChar(c)) {
			size_t start = i;
			while (i < expr.size() && isIdentChar(expr[i])) i++;
			std::string name = expr.substr(start, i - start);
			Token t;
			if (isFuncName(name)) {
				t.type = Token::Type::Func;
				t.text = name;
				t.arity = funcArity(name);
			} else if (name == "pi" || name == "PI") {
				t.type = Token::Type::Number;
				t.value = 3.14159265358979323846;
			} else if (name == "e" || name == "E") {
				t.type = Token::Type::Number;
				t.value = 2.71828182845904523536;
			} else {
				ok = false;
				return {};
			}
			out.push_back(t);
			hasPrev = true; prev = t;
			continue;
		}
		if (c == '(') {
			Token t; t.type = Token::Type::LParen; t.text = "("; out.push_back(t);
			hasPrev = true; prev = t; i++; continue;
		}
		if (c == ')') {
			Token t; t.type = Token::Type::RParen; t.text = ")"; out.push_back(t);
			hasPrev = true; prev = t; i++; continue;
		}
		if (c == ',') {
			Token t; t.type = Token::Type::Comma; t.text = ","; out.push_back(t);
			hasPrev = true; prev = t; i++; continue;
		}
		if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
			Token t; t.type = Token::Type::Op; t.text = std::string(1, c);
			bool unary = false;
			if (!hasPrev || prev.type == Token::Type::Op || prev.type == Token::Type::LParen || prev.type == Token::Type::Comma) {
				unary = (c == '-');
			}
			if (unary) {
				t.text = "~";
				t.precedence = 4;
				t.rightAssoc = true;
				t.arity = 1;
			} else if (c == '+' || c == '-') {
				t.precedence = 1;
				t.rightAssoc = false;
				t.arity = 2;
			} else if (c == '*' || c == '/') {
				t.precedence = 2;
				t.rightAssoc = false;
				t.arity = 2;
			} else if (c == '^') {
				t.precedence = 3;
				t.rightAssoc = true;
				t.arity = 2;
			}
			out.push_back(t);
			hasPrev = true; prev = t; i++; continue;
		}
		ok = false;
		return {};
	}
	return out;
}

static std::vector<Token> toRpn(const std::vector<Token> &tokens, bool &ok) {
	ok = true;
	std::vector<Token> output;
	std::vector<Token> ops;
	for (const auto &tok : tokens) {
		if (tok.type == Token::Type::Number) {
			output.push_back(tok);
			continue;
		}
		if (tok.type == Token::Type::Func) {
			ops.push_back(tok);
			continue;
		}
		if (tok.type == Token::Type::Comma) {
			while (!ops.empty() && ops.back().type != Token::Type::LParen) {
				output.push_back(ops.back());
				ops.pop_back();
			}
			if (ops.empty()) { ok = false; return {}; }
			continue;
		}
		if (tok.type == Token::Type::Op) {
			while (!ops.empty()) {
				const auto &top = ops.back();
				if (top.type != Token::Type::Op) break;
				if ((tok.rightAssoc && tok.precedence < top.precedence) || (!tok.rightAssoc && tok.precedence <= top.precedence)) {
					output.push_back(top);
					ops.pop_back();
				} else {
					break;
				}
			}
			ops.push_back(tok);
			continue;
		}
		if (tok.type == Token::Type::LParen) { ops.push_back(tok); continue; }
		if (tok.type == Token::Type::RParen) {
			while (!ops.empty() && ops.back().type != Token::Type::LParen) {
				output.push_back(ops.back());
				ops.pop_back();
			}
			if (ops.empty()) { ok = false; return {}; }
			ops.pop_back();
			if (!ops.empty() && ops.back().type == Token::Type::Func) {
				output.push_back(ops.back());
				ops.pop_back();
			}
			continue;
		}
		ok = false;
		return {};
	}
	while (!ops.empty()) {
		if (ops.back().type == Token::Type::LParen || ops.back().type == Token::Type::RParen) {
			ok = false;
			return {};
		}
		output.push_back(ops.back());
		ops.pop_back();
	}
	return output;
}

static double evalRpn(const std::vector<Token> &rpn, bool &ok) {
	ok = true;
	std::vector<double> st;
	for (const auto &tok : rpn) {
		if (tok.type == Token::Type::Number) {
			st.push_back(tok.value);
			continue;
		}
		if (tok.type == Token::Type::Op) {
			if (tok.arity == 1) {
				if (st.empty()) { ok = false; return 0.0; }
				double a = st.back(); st.pop_back();
				st.push_back(-a);
				continue;
			}
			if (st.size() < 2) { ok = false; return 0.0; }
			double b = st.back(); st.pop_back();
			double a = st.back(); st.pop_back();
			double v = 0.0;
			if (tok.text == "+") v = a + b;
			else if (tok.text == "-") v = a - b;
			else if (tok.text == "*") v = a * b;
			else if (tok.text == "/") v = b == 0.0 ? std::numeric_limits<double>::quiet_NaN() : a / b;
			else if (tok.text == "^") v = std::pow(a, b);
			else { ok = false; return 0.0; }
			st.push_back(v);
			continue;
		}
		if (tok.type == Token::Type::Func) {
			if (tok.arity == 1) {
				if (st.empty()) { ok = false; return 0.0; }
				double a = st.back(); st.pop_back();
				double v = 0.0;
				if (tok.text == "sin") v = std::sin(a);
				else if (tok.text == "cos") v = std::cos(a);
				else if (tok.text == "tan") v = std::tan(a);
				else if (tok.text == "tanh") v = std::tanh(a);
				else if (tok.text == "exp") v = std::exp(a);
				else if (tok.text == "log") v = std::log(a);
				else if (tok.text == "sqrt") v = a < 0.0 ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(a);
				else if (tok.text == "abs") v = std::fabs(a);
				else { ok = false; return 0.0; }
				st.push_back(v);
				continue;
			}
			if (st.size() < 2) { ok = false; return 0.0; }
			double b = st.back(); st.pop_back();
			double a = st.back(); st.pop_back();
			double v = 0.0;
			if (tok.text == "min") v = std::min(a, b);
			else if (tok.text == "max") v = std::max(a, b);
			else if (tok.text == "pow") v = std::pow(a, b);
			else { ok = false; return 0.0; }
			st.push_back(v);
			continue;
		}
		ok = false;
		return 0.0;
	}
	if (st.size() != 1 || !std::isfinite(st.back())) { ok = false; return 0.0; }
	return st.back();
}

static bool looksLikeExpr(const std::string &text) {
	std::string t = text;
	auto trim = [&](std::string s) {
		auto start = s.find_first_not_of(" \t\r\n");
		if (start == std::string::npos) return std::string();
		auto end = s.find_last_not_of(" \t\r\n");
		return s.substr(start, end - start + 1);
	};
	t = trim(t);
	if (t.empty()) return false;
	bool hasOp = false;
	for (char c : t) {
		if (std::isdigit(static_cast<unsigned char>(c))) continue;
		if (std::isspace(static_cast<unsigned char>(c))) continue;
		if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '(' || c == ')' || c == '.' || c == ',') { hasOp = true; continue; }
		if (isIdentChar(c)) { hasOp = true; continue; }
		return false;
	}
	return hasOp;
}

static std::string extractExpr(const std::string &text) {
	std::string t = text;
	std::string lower = t;
	std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	std::vector<std::string> prefixes = {"calc:", "math:", "计算:", "求值:", "calc=", "math=", "计算=", "求值="};
	for (const auto &p : prefixes) {
		if (lower.rfind(p, 0) == 0) return t.substr(p.size());
	}
	return t;
}

static std::string trimCopy(const std::string &text) {
	auto start = text.find_first_not_of(" \t\r\n");
	if (start == std::string::npos) return std::string();
	auto end = text.find_last_not_of(" \t\r\n");
	return text.substr(start, end - start + 1);
}

static std::optional<std::string> runCalculatorBridge(const std::string &expr) {
#ifdef _WIN32
	namespace fs = std::filesystem;
	static std::mutex cacheMu;
	static std::unordered_map<std::string, std::string> cache;
	{
		std::lock_guard<std::mutex> lock(cacheMu);
		auto it = cache.find(expr);
		if (it != cache.end()) return it->second;
	}
	fs::path exe = fs::current_path() / "outsides" / "_calculator" / "main.exe";
	if (!fs::exists(exe)) return std::nullopt;
	std::string command = "cmd /C \"(echo " + expr + "& echo exit) | \"\"" + exe.string() + "\"\"\"";
	FILE *pipe = _popen(command.c_str(), "r");
	if (!pipe) return std::nullopt;
	std::string output;
	char buffer[512];
	while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
	_pclose(pipe);
	std::istringstream iss(output);
	std::string line;
	std::string last;
	while (std::getline(iss, line)) {
		line = trimCopy(line);
		if (!line.empty()) last = line;
	}
	if (last.empty()) return std::nullopt;
	{
		std::lock_guard<std::mutex> lock(cacheMu);
		cache[expr] = last;
	}
	return last;
#else
	(void)expr;
	return std::nullopt;
#endif
}

class MathAddon : public Addon {
public:
	explicit MathAddon(std::string name) : name_(std::move(name)) {}
	std::string name() const override { return name_; }
	std::string type() const override { return "math"; }
	AddonResult handle(const std::string &text, const json &payload) override {
		AddonResult res;
		std::string expr = extractExpr(text);
		bool preferExternal = payload.value("mathBridge", std::string()) == "calculator";
		if (!preferExternal) {
			std::string lower = text;
			std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char)std::tolower(c); });
			preferExternal = lower.rfind("calculus ", 0) == 0 || lower.rfind("signal ", 0) == 0 || lower.rfind("brain:", 0) == 0;
		}
		if (preferExternal) {
			auto external = runCalculatorBridge(expr);
			if (external) {
				res.handled = true;
				res.reply = "结果: " + *external;
				res.meta = json{{"addon", "math"}, {"name", name_}, {"expression", expr}, {"bridge", "calculator"}, {"brainProjection", json{{"semanticBands", 4}, {"windowTokens", 1024}}}};
				return res;
			}
		}
		if (!looksLikeExpr(expr)) return res;
		bool ok = true;
		auto tokens = tokenizeExpr(expr, ok);
		if (!ok) return res;
		auto rpn = toRpn(tokens, ok);
		if (!ok) return res;
		double v = evalRpn(rpn, ok);
		if (!ok || !std::isfinite(v)) return res;
		std::ostringstream oss;
		oss.setf(std::ios::fixed);
		oss.precision(10);
		oss << v;
		std::string value = oss.str();
		value.erase(value.find_last_not_of('0') + 1);
		if (!value.empty() && value.back() == '.') value.pop_back();
		res.handled = true;
		res.reply = "结果: " + value;
		res.meta = json{{"addon", "math"}, {"name", name_}, {"expression", expr}, {"value", v}, {"bridge", "builtin"}, {"brainProjection", json{{"semanticBands", 2}, {"windowTokens", 512}}}};
		return res;
	}

private:
	std::string name_;
};

} // namespace

std::shared_ptr<Addon> createMathAddon(const std::string &name) {
	const std::string addonName = name.empty() ? std::string("math") : name;
	return std::make_shared<MathAddon>(addonName);
}

} // namespace addon::builtins
