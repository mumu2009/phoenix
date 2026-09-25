#include "memetic_existence.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL " << #cond << " line " << __LINE__ << "\n";           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  using namespace phoenix::memetic;
  InMemoryStore store;
  const std::string corpus =
      "The south river survey marker stands beside the tidal marsh after "
      "harvest season and records the channel depth for local boats. "
      "A lunar greenhouse harvest schedule lists polar survey beacon grid "
      "times for the winter crew. The bakery inventory shelf count lists "
      "flour bags and yeast jars for the morning shift only.";
  CHECK(ingestCorpusText(store, corpus, 8) >= 2);
  const size_t before = store.nodeCount();
  const auto screened = screenMemesFromBarrier(store, 6);
  CHECK(!screened.empty());
  const auto sample = screened.front();
  CHECK(sample.carrier == carrierOfMeme(store, sample.id));
  CHECK(detectMemeInText(store, sample.carrier, sample.id).present);
  CHECK(store.nodeCount() == before);
  CHECK(!detectMemeInText(store, "quantum chess tournament pairing sheet omega7",
                          sample.id)
             .present);
  CHECK(store.nodeCount() == before);

  InMemoryStore store2;
  ingestDocumentLike(store2, {"south", "river", "survey", "marker", "tealridge08"},
                     {});
  const std::string forced = expressActivated(
      store2, {"south", "river", "survey", "marker", "tealridge08"}, {}, 0,
      "tealridge08");
  CHECK(forced.find("tealridge08") != std::string::npos);
  CHECK(forced.find("noted ") != std::string::npos);
  CHECK(carrierOfMeme(store2, store2.nodeIds().front()).find("noted ") ==
        std::string::npos);

  InMemoryStore exclusive;
  exclusive.ensureNode("meme_p_target");
  exclusive.ensureNode("meme_p_other");
  for (const char *w : {"tealridge08", "river", "survey", "marker", "channel"})
    exclusive.bind(w, "meme_p_target");
  for (const char *w : {"bakery", "festival", "harbor", "cakes", "lunar"})
    exclusive.bind(w, "meme_p_other");
  for (int i = 0; i < 4; ++i) {
    const std::string id = "meme_p_decoy" + std::to_string(i);
    exclusive.ensureNode(id);
    exclusive.bind(std::string("decoy") + char('w' + i) + "word", id);
  }
  const std::string exclusiveSent =
      "Tealridge08 river survey marker records the south channel depth after "
      "harvest time.";
  const std::string mixedSent =
      "Tealridge08 bakery festival cakes at the harbor lunar greenhouse after "
      "harvest season were sold beside the river survey marker channel.";
  const auto rag = composeCarrierRag(exclusive, "meme_p_target",
                                     exclusiveSent + " " + mixedSent, 8, 2);
  CHECK(!rag.text.empty());
  CHECK(rag.text.find("Tealridge08") != std::string::npos);
  CHECK(rag.text.find("lunar") == std::string::npos);
  CHECK(rag.otherPresent <= 0);
  CHECK(rag.unitIndex.size() <= 1);
  CHECK(!rag.sources.empty());
  CHECK(detectMemeInText(exclusive, rag.text, "meme_p_target").present);

  InMemoryStore named;
  named.ensureNode("meme_p_lock");
  named.ensureNode("meme_p_other");
  for (const char *w : {"captain", "green", "lockman", "company", "patrol"})
    named.bind(w, "meme_p_lock");
  for (const char *w : {"bakery", "festival", "harbor", "cakes", "lunar"})
    named.bind(w, "meme_p_other");
  for (int i = 0; i < 4; ++i) {
    const std::string id = "meme_p_decoy" + std::to_string(i);
    named.ensureNode(id);
    named.bind(std::string("decoy") + char('w' + i) + "word", id);
  }
  const std::string lockSent =
      "Captain Lockman and lieutenant Green held the company line after dawn "
      "patrol ended.";
  const std::string greenOnly =
      "Jordan father was murdered at a highway rest area by two teenagers "
      "Daniel Green and Larry Martin Demery later that night.";
  const auto namedRag =
      composeCarrierRag(named, "meme_p_lock", lockSent + " " + greenOnly, 8, 1);
  CHECK(!namedRag.text.empty());
  CHECK(namedRag.text.find("Lockman") != std::string::npos);
  CHECK(namedRag.text.find("Demery") == std::string::npos);

  const std::string snap = "build/tmp_gnn_instrument_test.txt";
  store.saveText(snap);
  InMemoryStore loaded;
  CHECK(loaded.loadText(snap));
  CHECK(detectMemeInText(loaded, sample.carrier, sample.id).present);
  std::cout << "ok screened=" << screened.size() << " id=" << sample.id << "\n";
  return 0;
}
