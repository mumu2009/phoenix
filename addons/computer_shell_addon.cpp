/* computer_shell_addon.cpp - Computer shell addon implementation
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

#include "computer_shell_addon.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace addon::builtins {

namespace {

std::string trimCopy(const std::string &value) {
	auto start = value.find_first_not_of(" \t\r\n");
	if (start == std::string::npos) return "";
	auto end = value.find_last_not_of(" \t\r\n");
	return value.substr(start, end - start + 1);
}

std::string lowerCopy(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return value;
}

bool isComputerAlias(const std::string &value) {
	const std::string lowered = lowerCopy(trimCopy(value));
	return lowered == "computer" || lowered == "shell" || lowered == "desktop";
}

std::string stripPrefix(const std::string &text) {
	std::string trimmed = trimCopy(text);
	std::string lowered = lowerCopy(trimmed);
	const std::vector<std::string> prefixes = {
		"computer:", "shell:", "desktop:", "电脑:", "桌面:"
	};
	for (const auto &prefix : prefixes) {
		if (lowered.rfind(prefix, 0) == 0) {
			return trimCopy(trimmed.substr(prefix.size()));
		}
	}
	return trimmed;
}

bool parseComputerRequest(const std::string &text, addon::json &request, std::string &error) {
	std::string normalized = stripPrefix(text);
	if (normalized.empty()) {
		request = addon::json{{"op", "status"}};
		return true;
	}
	if (normalized.front() == '{' || normalized.front() == '[') {
		auto parsed = addon::json::parse(normalized, nullptr, false);
		if (parsed.is_discarded() || !parsed.is_object()) {
			error = "computer shell payload must be a JSON object";
			return false;
		}
		request = parsed;
		return true;
	}

	std::string lowered = lowerCopy(normalized);
	if (lowered == "status" || lowered == "pwd") {
		request = addon::json{{"op", lowered == "pwd" ? "pwd" : "status"}};
		return true;
	}
	if (lowered.rfind("list ", 0) == 0 || lowered.rfind("dir ", 0) == 0 || lowered.rfind("ls ", 0) == 0) {
		std::size_t pos = lowered.find(' ');
		request = addon::json{{"op", "list_dir"}, {"path", trimCopy(normalized.substr(pos + 1))}};
		return true;
	}
	if (lowered.rfind("read ", 0) == 0 || lowered.rfind("cat ", 0) == 0 || lowered.rfind("type ", 0) == 0) {
		std::size_t pos = lowered.find(' ');
		request = addon::json{{"op", "read_text"}, {"path", trimCopy(normalized.substr(pos + 1))}};
		return true;
	}
	if (lowered.rfind("open-url ", 0) == 0) {
		request = addon::json{{"op", "open_url"}, {"url", trimCopy(normalized.substr(9))}};
		return true;
	}
	if (lowered.rfind("open ", 0) == 0) {
		request = addon::json{{"op", "open_path"}, {"path", trimCopy(normalized.substr(5))}};
		return true;
	}
	request = addon::json{{"op", "run"}, {"command", normalized}, {"shell", true}};
	return true;
}

std::string summarizeBridgeResult(const addon::json &bridge) {
	if (!bridge.is_object()) return "computer shell bridge returned invalid result";
	if (bridge.contains("summary") && bridge["summary"].is_string()) {
		return bridge["summary"].get<std::string>();
	}
	if (!bridge.value("ok", false)) {
		std::string error = trimCopy(bridge.value("error", std::string("computer shell bridge failed")));
		return error.empty() ? std::string("computer shell bridge failed") : error;
	}
	if (bridge.contains("stdout") && bridge["stdout"].is_string()) {
		std::string stdoutText = trimCopy(bridge["stdout"].get<std::string>());
		if (!stdoutText.empty()) return stdoutText;
	}
	if (bridge.contains("content") && bridge["content"].is_string()) {
		std::string content = trimCopy(bridge["content"].get<std::string>());
		if (!content.empty()) return content;
	}
	if (bridge.contains("entries") && bridge["entries"].is_array()) {
		std::ostringstream oss;
		oss << "listed " << bridge["entries"].size() << " entries";
		if (bridge.contains("path") && bridge["path"].is_string()) {
			oss << " in " << bridge["path"].get<std::string>();
		}
		return oss.str();
	}
	return bridge.dump();
}

class ComputerShellAddon : public Addon {
public:
	explicit ComputerShellAddon(std::string name) : name_(std::move(name)) {}

	std::string name() const override { return name_; }
	std::string type() const override { return "computer"; }

	AddonResult handle(const std::string &text, const json &payload) override {
		AddonResult result;
		std::string addonType = trimCopy(payload.value("__addonType", std::string()));
		if (!addonType.empty() && !isComputerAlias(addonType)) return result;

		json request;
		std::string parseError;
		if (!parseComputerRequest(text, request, parseError)) {
			result.handled = true;
			result.reply = std::string("computer shell request error: ") + parseError;
			result.meta = json{{"addon", "computer"}, {"name", name_}, {"ok", false}, {"error", parseError}};
			return result;
		}

		json options = payload.contains("computerShellOptions") && payload["computerShellOptions"].is_object()
			? payload["computerShellOptions"]
			: json::object();
		json bridge;
		bool ok = addon::invokeAddonComputerShell(request, options, bridge);
		result.handled = true;
		if (!ok && bridge.is_null()) {
			bridge = json{{"ok", false}, {"error", "computer shell bridge unavailable"}};
		}
		result.reply = summarizeBridgeResult(bridge);
		if (result.reply.size() > 4000) result.reply.resize(4000);
		result.meta = json{{"addon", "computer"},
						 {"name", name_},
						 {"ok", bridge.value("ok", ok)},
						 {"op", request.value("op", std::string("status"))}};
		if (bridge.contains("error") && bridge["error"].is_string()) {
			result.meta["error"] = bridge["error"].get<std::string>();
		}
		return result;
	}

private:
	std::string name_;
};

} // namespace

std::shared_ptr<Addon> createComputerShellAddon(const std::string &name) {
	const std::string addonName = name.empty() ? std::string("computer") : name;
	return std::make_shared<ComputerShellAddon>(addonName);
}

} // namespace addon::builtins