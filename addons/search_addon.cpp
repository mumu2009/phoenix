/* search_addon.cpp - Search addon implementation
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

#include "search_addon.hpp"

#include <algorithm>
#include <cctype>

namespace addon::builtins {

namespace {

static std::string trimCopy(const std::string &s) {
	auto start = s.find_first_not_of(" \t\r\n");
	if (start == std::string::npos) return "";
	auto end = s.find_last_not_of(" \t\r\n");
	return s.substr(start, end - start + 1);
}

static std::string lowerCopy(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	return s;
}

static std::string stripPrefix(const std::string &text) {
	std::string t = trimCopy(text);
	std::string l = lowerCopy(t);
	const std::vector<std::string> prefixes = {
		"search:", "web:", "lookup:", "research:", "搜索:", "检索:", "查询:"
	};
	for (const auto &p : prefixes) {
		if (l.rfind(p, 0) == 0) {
			return trimCopy(t.substr(p.size()));
		}
	}
	return t;
}

class SearchAddon : public Addon {
public:
	explicit SearchAddon(std::string name) : name_(std::move(name)) {}
	std::string name() const override { return name_; }
	std::string type() const override { return "search"; }

	AddonResult handle(const std::string &text, const json &payload) override {
		AddonResult res;
		std::string addonType = lowerCopy(trimCopy(payload.value("__addonType", std::string())));
		bool explicitSearch = (addonType == "search" || addonType == "research" || addonType == "web");
		if (!explicitSearch && !addonType.empty()) return res;

		std::string query = stripPrefix(text);
		if (query.empty()) return res;

		json options = payload.contains("searchOptions") && payload["searchOptions"].is_object()
			? payload["searchOptions"]
			: json::object();
		if (!options.contains("preferIndex")) options["preferIndex"] = true;

		json lookup;
		bool ok = addon::invokeAddonOnlineLookup(json(query), options, lookup);
		if (!ok || lookup.is_null()) return res;

		std::string reply;
		if (lookup.contains("snippet") && lookup["snippet"].is_string()) {
			reply = lookup["snippet"].get<std::string>();
		} else if (lookup.contains("text") && lookup["text"].is_string()) {
			reply = lookup["text"].get<std::string>();
		} else if (lookup.contains("suggestions") && lookup["suggestions"].is_array() && !lookup["suggestions"].empty()) {
			const auto &first = lookup["suggestions"][0];
			if (first.is_object() && first.contains("words") && first["words"].is_array()) {
				std::string joined;
				for (size_t i = 0; i < first["words"].size(); ++i) {
					if (i) joined += " ";
					joined += first["words"][i].is_string() ? first["words"][i].get<std::string>() : first["words"][i].dump();
				}
				reply = joined;
			}
		}
		if (reply.empty()) reply = lookup.dump();
		if (reply.size() > 1200) reply.resize(1200);

		res.handled = true;
		res.reply = reply;
		res.meta = json{{"addon", "search"}, {"name", name_}, {"query", query}, {"source", lookup.value("source", "unknown")}};
		return res;
	}

private:
	std::string name_;
};

} // namespace

std::shared_ptr<Addon> createSearchAddon(const std::string &name) {
	const std::string addonName = name.empty() ? std::string("search") : name;
	return std::make_shared<SearchAddon>(addonName);
}

} // namespace addon::builtins
