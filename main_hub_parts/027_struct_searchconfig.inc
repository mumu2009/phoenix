/* Auto-generated from main.cpp
   Segment: struct_SearchConfig
   Original lines: 3448-3457 */

struct SearchConfig {
  bool enabled{false};
  std::vector<std::string> endpoints;
  std::string active;

  json toJson() const {
    return json{
        {"enabled", enabled}, {"endpoints", endpoints}, {"active", active}};
  }
};
