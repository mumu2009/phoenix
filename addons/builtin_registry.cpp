/* builtin_registry.cpp - Builtin addon registry implementation
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

#include "builtin_registry.hpp"

#include "cli_json_addon.hpp"
#include "computer_shell_addon.hpp"
#include "math_addon.hpp"
#include "search_addon.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace addon::builtins {

namespace {

std::string lowerCopy(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return value;
}

std::unordered_set<std::string> parseSelection(const std::string &selection) {
	std::unordered_set<std::string> out;
	std::string normalized = selection;
	for (char &ch : normalized) {
		if (ch == ';' || ch == '\n' || ch == '\r') {
			ch = ',';
		}
	}
	std::string current;
	for (char ch : normalized) {
		if (ch == ',') {
			if (!current.empty()) {
				out.insert(lowerCopy(current));
				current.clear();
			}
			continue;
		}
		if (!std::isspace(static_cast<unsigned char>(ch))) {
			current.push_back(ch);
		}
	}
	if (!current.empty()) {
		out.insert(lowerCopy(current));
	}
	return out;
}

bool wantsAddon(const std::unordered_set<std::string> &selected, const std::string &name) {
	if (selected.empty()) {
		return true;
	}
	if (selected.count("all") || selected.count("default")) {
		return true;
	}
	if (selected.count("none")) {
		return false;
	}
	return selected.count(lowerCopy(name)) > 0;
}

} // namespace

std::shared_ptr<Addon> createBuiltinAddon(const std::string &type, const std::string &name, std::string *error) {
	if (type == "math") {
		return createMathAddon(name.empty() ? std::string("math") : name);
	}
	if (type == "search") {
		return createSearchAddon(name.empty() ? std::string("search") : name);
	}
	if (type == "computer" || type == "shell" || type == "desktop") {
		return createComputerShellAddon(name.empty() ? std::string("computer") : name);
	}
	if (error) *error = "unsupported addon type";
	return nullptr;
}

std::vector<std::shared_ptr<Addon>> createDefaultBuiltinAddons(const std::string &selection) {
	std::vector<std::shared_ptr<Addon>> out;
	const auto selected = parseSelection(selection);
	if (wantsAddon(selected, "math")) {
		out.push_back(createMathAddon("math"));
	}
	if (wantsAddon(selected, "search")) {
		out.push_back(createSearchAddon("search"));
	}
	if (wantsAddon(selected, "computer")) {
		out.push_back(createComputerShellAddon("computer"));
	}
	return out;
}

} // namespace addon::builtins

namespace addon {

std::shared_ptr<AddonManager> createDefaultAddons(const std::string &builtinSelection) {
	auto mgr = std::make_shared<AddonManager>();
	for (auto &builtinAddon : builtins::createDefaultBuiltinAddons(builtinSelection)) {
		mgr->registerAddon(builtinAddon);
	}
	return mgr;
}

} // namespace addon
