/* cli_json_addon.cpp - Implementation, see header.
   This is the Phoenix-side counterpart of the CLI-Anything idea (see
   outsides/CLI-Anything): any CLI program that can emit --json output becomes
   a plugin by registering ONE whitelist entry.  The harness stays in-process;
   the command runs via phoenix::subprocess (direct exec, no shell). */
#include "cli_json_addon.hpp"

#include "../subprocess.hpp"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <sstream>
#include <vector>

namespace addon::builtins {

namespace {

struct CliTemplate {
  std::string command;
  std::vector<std::string> fixedArgs;
  int timeoutMs{5000};
  bool jsonOutput{true};
  size_t maxReply{2000};
};

std::mutex gRegistryMu;
std::map<std::string, CliTemplate> gRegistry;

std::string lowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return s;
}

/* tokenize request text into argv pieces (whitespace split, quotes kept
   literal - no shell semantics). */
std::vector<std::string> splitArgs(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : text) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

} /* namespace */

bool setCliJsonRegistry(const json &registry, std::string *error) {
  std::lock_guard<std::mutex> lock(gRegistryMu);
  std::map<std::string, CliTemplate> next;
  if (!registry.is_array()) {
    if (error) *error = "cliTools registry must be an array";
    return false;
  }
  for (const auto &item : registry) {
    if (!item.is_object()) continue;
    const std::string name = item.value("name", std::string());
    CliTemplate tpl;
    tpl.command = item.value("command", std::string());
    if (item.contains("args") && item["args"].is_array())
      for (const auto &a : item["args"]) if (a.is_string()) tpl.fixedArgs.push_back(a.get<std::string>());
    tpl.timeoutMs = item.value("timeoutMs", 5000);
    tpl.jsonOutput = item.value("json", item.value("jsonOutput", true));
    tpl.maxReply = item.value("maxReply", 2000);
    if (name.empty() || tpl.command.empty()) continue;
    next[name] = tpl;
  }
  gRegistry = std::move(next);
  return true;
}

void clearCliJsonRegistry() {
  std::lock_guard<std::mutex> lock(gRegistryMu);
  gRegistry.clear();
}

json getCliJsonRegistry() {
  std::lock_guard<std::mutex> lock(gRegistryMu);
  json out = json::array();
  for (const auto &kv : gRegistry) {
    out.push_back(json{{"name", kv.first}, {"command", kv.second.command},
                       {"args", kv.second.fixedArgs}, {"timeoutMs", kv.second.timeoutMs},
                       {"json", kv.second.jsonOutput}, {"maxReply", kv.second.maxReply}});
  }
  return out;
}

json runCliJsonCommand(const std::string &tool, const std::vector<std::string> &args,
                       const json &options) {
  CliTemplate tpl;
  {
    std::lock_guard<std::mutex> lock(gRegistryMu);
    auto it = gRegistry.find(tool);
    if (it == gRegistry.end()) {
      return json{{"ok", false}, {"error", "cli tool not whitelisted: " + tool}};
    }
    tpl = it->second;
  }
  phoenix::subprocess::RunRequest req;
  req.command = tpl.command;
  req.args = tpl.fixedArgs;
  for (const auto &a : args) req.args.push_back(a);
  req.timeoutMs = options.value("timeoutMs", tpl.timeoutMs);
  req.maxOutputBytes = 4u * 1024u * 1024u;
  const phoenix::subprocess::RunResult r = phoenix::subprocess::run(req);
  json out;
  out["tool"] = tool;
  out["exitCode"] = r.exitCode;
  out["timedOut"] = r.timedOut;
  out["ok"] = r.started && r.exitCode == 0 && !r.timedOut;
  if (!r.error.empty()) out["error"] = r.error;
  std::string reply;
  if (tpl.jsonOutput) {
    try {
      json parsed = json::parse(r.stdoutText);
      out["json"] = parsed;
      reply = parsed.dump();
    } catch (...) {
      /* fall back to text; still report raw */
      out["text"] = r.stdoutText;
      reply = r.stdoutText;
    }
  } else {
    out["text"] = r.stdoutText;
    reply = r.stdoutText;
  }
  if (reply.size() > tpl.maxReply) reply.resize(tpl.maxReply);
  out["reply"] = reply;
  if (!r.stderrText.empty()) out["stderr"] = r.stderrText.substr(0, 400);
  return out;
}

namespace {

class CliJsonAddon : public Addon {
public:
  explicit CliJsonAddon(std::string name) : name_(std::move(name)) {}
  std::string name() const override { return name_; }
  std::string type() const override { return "cli-json"; }

  AddonResult handle(const std::string &text, const json &payload) override {
    AddonResult res;
    std::string addonType = payload.value("__addonType", std::string());
    std::string lowType = lowerCopy(addonType);
    if (!lowType.empty() && lowType != "cli-json" && lowType != "cli" &&
        lowType != "clijson") return res;
    /* tool selection: payload["__cliTool"] or first word after "cli:" prefix */
    std::string tool = payload.value("__cliTool", std::string());
    std::string rest = text;
    if (tool.empty()) {
      std::string low = lowerCopy(text);
      if (low.rfind("cli:", 0) == 0) rest = text.substr(4);
      const auto parts = splitArgs(rest);
      if (parts.empty()) return res;
      tool = parts[0];
      rest.clear();
      for (size_t i = 1; i < parts.size(); ++i) {
        if (i > 1) rest += " ";
        rest += parts[i];
      }
    }
    if (tool.empty()) return res;
    json out = runCliJsonCommand(tool, splitArgs(rest), payload);
    res.handled = true;
    res.meta = json{{"addon", "cli-json"}, {"name", name_}, {"result", out}};
    res.reply = out.value("reply", std::string());
    if (!out.value("ok", false)) {
      res.reply = "[cli-json error] " + out.value("error", std::string("command failed"));
    }
    return res;
  }

private:
  std::string name_;
};

} /* namespace */

std::shared_ptr<Addon> createCliJsonAddon(const std::string &name) {
  const std::string addonName = name.empty() ? std::string("cli-json") : name;
  return std::make_shared<CliJsonAddon>(addonName);
}

} /* namespace addon::builtins */
