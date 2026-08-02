#include "component_config.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using OverrideMap = component_config::OverrideMap;

void requireTrue(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireValue(const OverrideMap &overrides, const std::string &key, const std::string &expected) {
    const auto it = overrides.find(key);
    if (it == overrides.end()) {
        throw std::runtime_error("missing override: " + key);
    }
    if (it->second != expected) {
        throw std::runtime_error("unexpected override value for " + key + ": " + it->second + " != " + expected);
    }
}

void testParseCliSpecSupportsFlatAliases() {
    const auto overrides = component_config::parseCliSpec("brain=structural,gnn=off,frontend-memory=on,collective-matching=storage-only");

    requireValue(overrides, "brain", "structural");
    requireValue(overrides, "gnn", "off");
    requireValue(overrides, "frontend-memory", "on");
    requireValue(overrides, "collective-matching", "storage-only");
}

void testParseJsonTextFlattensNestedSelections() {
    const std::string raw = R"JSON({
        "pipeline": {
            "gnn": false,
            "brain": {
                "enabled": true,
                "profile": "structural"
            }
        },
        "collective": {
            "routingStorage": true,
            "ignoreMatchingForComputeNodes": true
        }
    })JSON";

    const auto overrides = component_config::parseJsonText(raw);

    requireValue(overrides, "pipeline.gnn", "false");
    requireValue(overrides, "pipeline.brain.enabled", "true");
    requireValue(overrides, "pipeline.brain.profile", "structural");
    requireValue(overrides, "collective.routingstorage", "true");
    requireValue(overrides, "collective.ignorematchingforcomputenodes", "true");
}

void testParseXmlTextSupportsComponentPathEntries() {
    const std::string raw = R"XML(
<components>
  <component path="pipeline.gnn" enabled="false" />
  <component path="pipeline.brain.profile" value="functional" />
  <component path="collective.matching" value="storage-only" />
</components>
)XML";

    const auto overrides = component_config::parseXmlText(raw);

    requireValue(overrides, "pipeline.gnn", "false");
    requireValue(overrides, "pipeline.brain.profile", "functional");
    requireValue(overrides, "collective.matching", "storage-only");
}

} // namespace

int main() {
    try {
        testParseCliSpecSupportsFlatAliases();
        testParseJsonTextFlattensNestedSelections();
        testParseXmlTextSupportsComponentPathEntries();
        std::cout << "component_config_tests: ok" << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "component_config_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}