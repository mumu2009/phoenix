#include "controller_layout.hpp"

#include <stdexcept>
#include <string>

static void require(bool ok, const std::string &msg) {
  if (!ok)
    throw std::runtime_error(msg);
}

int main() {
  auto one = phoenix::planControllerLayout(1, 3, 7);
  require(one.totalControllers == 1, "AI_COUNT=1 must stay 1");
  require(one.groupCount == 1, "AI_COUNT=1 must not open extra groups");

  auto product = phoenix::planControllerLayout(7, 3, 7);
  require(product.totalControllers == 7, "aiCount=7 must not create 21");
  require(product.groupCount == 1, "budget 7 fills one group of 7");

  phoenix::BoundedHitMap hits(4);
  for (int i = 0; i < 20; ++i)
    hits.increment("k" + std::to_string(i));
  require(hits.size() <= 4, "hit map must stay capped");
  return 0;
}
