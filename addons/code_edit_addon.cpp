/* code_edit_addon.cpp - plugin side of the source-edit pair.
 *
 * Expert side: a summoned programming model writes code.
 * Plugin side (this addon): recognises that code (fenced block, raw body,
 * or an existing patch) and normalises it to FILE/OLD/NEW, the form the
 * replica trace counts as a source edit.  It never writes a file; the
 * gateway applies the patch through its normal workspace path. */

#include "code_edit_addon.hpp"

#include "../code_edit_parser.hpp"

namespace addon::builtins {

namespace {
using json = nlohmann::json;

class CodeEditAddon : public Addon {
 public:
  explicit CodeEditAddon(std::string name) : name_(std::move(name)) {}
  std::string name() const override { return name_; }
  std::string type() const override { return "code-edit"; }

  float consider(const json &situation) const override {
    const std::string text = situation.value("text", std::string());
    if (text.empty()) return 0.f;
    if (phoenix::code_edit::hasPatchForm(text)) return 1.0f;
    if (text.find("```") != std::string::npos) return 0.9f;
    if (text.find("#include") != std::string::npos ||
        text.find("\ndef ") != std::string::npos ||
        text.find("\nvoid ") != std::string::npos)
      return 0.7f;
    return 0.f;
  }

  AddonResult contribute(const json &situation) override {
    AddonResult out;
    const std::string text = situation.value("text", std::string());
    const std::string path = situation.value("path", std::string());
    const std::string patch =
        phoenix::code_edit::extractCodeEdit(text, path);
    if (patch.empty()) return out;
    out.handled = true;
    out.reply = patch;
    out.meta = json{{"kind", "code-edit"},
                    {"path", path},
                    {"patch", true},
                    {"chars", patch.size()}};
    return out;
  }

  AddonResult handle(const std::string &text, const json &payload) override {
    json sit = payload.is_object() ? payload : json::object();
    sit["text"] = text;
    return contribute(sit);
  }

 private:
  std::string name_;
};
}  // namespace

std::shared_ptr<Addon> createCodeEditAddon(const std::string &name) {
  return std::make_shared<CodeEditAddon>(
      name.empty() ? std::string("code-edit") : name);
}

std::string normaliseCodeEdit(const std::string &reply,
                              const std::string &path) {
  return phoenix::code_edit::extractCodeEdit(reply, path);
}

}  // namespace addon::builtins
