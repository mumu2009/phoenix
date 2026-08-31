#include <gtest/gtest.h>

#include "inference_unit_pipeline.hpp"

using phoenix::inference::UnitQueryIO;
using phoenix::inference::appendUnitQueryFromJson;
using phoenix::inference::appendUnitQueriesFromJsonArray;

TEST(UnitQueryIO, PluginTextAndRowsAreTheSameProtocol) {
  std::vector<UnitQueryIO> dst;
  appendUnitQueryFromJson(dst, nlohmann::json("plain paragraph about delay."));
  ASSERT_EQ(dst.size(), 1u);
  EXPECT_EQ(dst[0].modality, "text");
  EXPECT_NE(dst[0].content.find("delay"), std::string::npos);

  nlohmann::json arr = nlohmann::json::array();
  arr.push_back({{"modality", "text"}, {"content", "second packet"}});
  arr.push_back({{"modality", "unit"},
                 {"rows", nlohmann::json::array({nlohmann::json::array({0.1, 0.2})})}});
  arr.push_back({{"modality", "image"}, {"content", "must-skip"}});
  appendUnitQueriesFromJsonArray(dst, arr);
  ASSERT_EQ(dst.size(), 3u);
  EXPECT_EQ(dst[1].content, "second packet");
  ASSERT_EQ(dst[2].rows.size(), 1u);
  EXPECT_NEAR(dst[2].rows[0][0], 0.1f, 1e-5);
}
