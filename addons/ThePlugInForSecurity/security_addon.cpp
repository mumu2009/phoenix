/* security_addon.cpp - Read-only security observatory addon */

#include "security_addon.hpp"
#include "security_core.hpp"

#include <algorithm>
#include <cctype>

namespace addon {
namespace builtins {

namespace {

class SecurityAddon final : public Addon {
public:
  explicit SecurityAddon(std::string name) : name_(std::move(name)) {
    phoenix::secamp::installSecurityModuleResources();
  }

  std::string name() const override { return name_; }
  std::string type() const override { return "security"; }

  float consider(const json &situation) const override {
    const std::string text = situation.value("text", std::string()) + " " +
                             situation.value("goal", std::string());
    std::string low = text;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (low.find("memebarrier") != std::string::npos ||
        low.find("gnn influence") != std::string::npos ||
        low.find("security stats") != std::string::npos)
      return 0.72f;
    return 0.f;
  }

  AddonResult handle(const std::string &text, const json &payload) override {
    (void)payload;
    phoenix::secamp::installSecurityModuleResources();
    AddonResult res;
    res.handled = true;
    const auto tokens = phoenix::secamp::tokenizeDefense(text);
    bool wantIdentify = false;
    bool wantAlerts = false;
    for (const auto &t : tokens) {
      if (t == "identify" || t == "mapping")
        wantIdentify = true;
      if (t == "alert" || t == "alerts")
        wantAlerts = true;
    }
    auto &obs = phoenix::secamp::SecurityObservatory::instance();
    if (wantAlerts)
      res.meta = obs.alertsJson();
    else if (wantIdentify)
      res.meta = obs.identifyListJson();
    else
      res.meta = obs.statsJson();
    res.reply = res.meta.dump();
    if (res.reply.size() > 2000)
      res.reply.resize(2000);
    sealAddonResultUnits(res);
    return res;
  }

private:
  std::string name_;
};

} // namespace

std::shared_ptr<Addon> createSecurityAddon(const std::string &name) {
  return std::make_shared<SecurityAddon>(name.empty() ? "security" : name);
}

} // namespace builtins
} // namespace addon
