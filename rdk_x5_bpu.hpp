#pragma once

#include <nlohmann/json.hpp>

namespace rdk_x5_bpu {

using json = nlohmann::json;

bool available();
bool requested(const json &payload);
json inspect(const json &payload);
json execute(const json &payload);

} // namespace rdk_x5_bpu
