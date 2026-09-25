/* Host gates for 1/df + graph-prior lift. No llama. */
#include "memetic_existence.hpp"

#include <cstdio>
#include <cstdlib>

using namespace phoenix::memetic;

static int gFails = 0;

static void expect(bool cond, const char *msg) {
  if (cond)
    return;
  ++gFails;
  std::fprintf(stderr, "FAIL %s\n", msg);
}

int main() {
  expect(mappingDegreeSeed(1.0, 1) == 1.0, "seed df=1");
  expect(mappingDegreeSeed(1.0, 400) == 1.0 / 400.0, "seed df=400");
  expect(mappingDegreeSeed(2.0, 0) == 0.0, "seed df=0");
  expect(mappingHighDfCut(497) == 23, "cut 497");
  expect(mappingHighDf(200, 497), "df 200 high");
  expect(mappingHighDf(24, 497), "df 24 high");
  expect(!mappingHighDf(23, 497), "df 23 not high");
  expect(!mappingHighDf(1, 497), "df 1 not high");

  InMemoryStore store;
  const char *ids[] = {"meme_p_hub", "meme_p_b", "meme_p_c",
                       "meme_p_d", "meme_p_e", "meme_p_f"};
  const char *uniq[] = {"river", "bakery", "quarry",
                        "orchard", "harbor", "depot"};
  for (int i = 0; i < 6; ++i) {
    store.bind("the", ids[i]);
    store.bind("of", ids[i]);
    store.bind("and", ids[i]);
    store.bind(uniq[i], ids[i]);
  }
  const auto prior = highDfPriorTokens(store);
  expect(!prior.empty(), "prior nonempty");
  const auto own = detectMemeInText(store, "river marker survey", ids[0]);
  std::printf("own present=%d lift=%.3f null=%.5f\n", own.present ? 1 : 0,
              own.lift, own.nullScore);
  expect(own.present, "own present");
  expect(own.lift >= kGnnPresentLift, "own lift");
  const auto stops = detectMemeInText(
      store, "the of and to a in on for is was with as by at from that this",
      ids[0]);
  std::printf("stop present=%d lift=%.3f act=%.5f\n", stops.present ? 1 : 0,
              stops.lift, stops.memeScore);
  expect(!stops.present, "stops absent");
  const auto miss = detectMemeInText(
      store, "quantum chess tournament pairing sheet omega7", ids[0]);
  expect(!miss.present, "unrelated absent");

  if (gFails > 0) {
    std::fprintf(stderr, "%d fails\n", gFails);
    return 1;
  }
  std::printf("ok prior=%zu ownLift=%.3f stopLift=%.3f\n", prior.size(),
              own.lift, stops.lift);
  return 0;
}
