/* v10.1 symbolic layer: the Prolog/Datalog-style reasoner.

   The rest of the stack is probabilistic (8B prose, GNN association, tiny
   expert scoring).  Nothing in it could answer "is this certainly true?".
   These tests pin the deterministic contract the mission tools rely on:
     - a Horn-clause closure reaches its fixpoint (transitive ancestor),
     - SLD resolution returns every solution, not just the first,
     - builtin comparisons gate a rule on ground numbers,
     - a cyclic rule base terminates inside the bounds instead of hanging,
     - the JSON façade parses the compact "pred(A,b)" form the model emits,
     - non-ground facts are rejected (fail closed). */
#include <gtest/gtest.h>

#include "logic_engine.hpp"

using phoenix::logic::Engine;
using phoenix::logic::json;

TEST(LogicEngine, Sample1_ForwardChainingDerivesTransitiveClosure) {
  Engine e;
  e.addFact({"parent", {phoenix::logic::Term::atom("tom"),
                        phoenix::logic::Term::atom("bob")}});
  e.addFact({"parent", {phoenix::logic::Term::atom("bob"),
                        phoenix::logic::Term::atom("ann")}});
  using phoenix::logic::Term;
  /* ancestor(X,Y) :- parent(X,Y) */
  e.addRule({{"ancestor", {Term::var("X"), Term::var("Y")}},
             {{"parent", {Term::var("X"), Term::var("Y")}}}});
  /* ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y) */
  e.addRule({{"ancestor", {Term::var("X"), Term::var("Y")}},
             {{"parent", {Term::var("X"), Term::var("Z")}},
              {"ancestor", {Term::var("Z"), Term::var("Y")}}}});
  const size_t derived = e.saturate();
  /* 2 parents + ancestor(tom,bob) + ancestor(bob,ann) + ancestor(tom,ann) */
  EXPECT_EQ(derived, 3u);
  EXPECT_EQ(e.factCount(), 5u);
}

TEST(LogicEngine, Sample2_BackwardChainingReturnsEverySolution) {
  const json req = json{
      {"facts", json::array({"parent(tom,bob)", "parent(bob,ann)"})},
      {"rules",
       json::array({json{{"head", "ancestor(X,Y)"},
                         {"body", json::array({"parent(X,Y)"})}},
                    json{{"head", "ancestor(X,Y)"},
                         {"body", json::array({"parent(X,Z)",
                                               "ancestor(Z,Y)"})}}})},
      {"query", "ancestor(tom,Who)"}};
  const json out = phoenix::logic::runJson(req);
  ASSERT_TRUE(out["ok"].get<bool>()) << out.dump();
  EXPECT_EQ(out["solutionCount"].get<int>(), 2);
}

TEST(LogicEngine, Sample3_BuiltinComparisonGatesRule) {
  const json req = json{
      {"facts", json::array({"age(alice,34)", "age(bob,12)"})},
      {"rules",
       json::array({json{{"head", "adult(X)"},
                         {"body", json::array({"age(X,A)", ">=(A,18)"})}}})},
      {"query", "adult(Who)"}};
  const json out = phoenix::logic::runJson(req);
  ASSERT_TRUE(out["ok"].get<bool>()) << out.dump();
  ASSERT_EQ(out["solutionCount"].get<int>(), 1);
  EXPECT_EQ(out["solutions"][0]["Who"].get<std::string>(), "alice");
}

TEST(LogicEngine, Sample4_CyclicRuleBaseTerminatesInsideBounds) {
  const json req = json{
      {"facts", json::array({"seed(a)"})},
      {"rules",
       json::array({json{{"head", "p(X)"}, {"body", json::array({"q(X)"})}},
                    json{{"head", "q(X)"}, {"body", json::array({"p(X)"})}}})},
      {"mode", "forward"},
      {"query", "p(a)"},
      {"maxIterations", 20}};
  const json out = phoenix::logic::runJson(req);
  ASSERT_TRUE(out["ok"].get<bool>()) << out.dump();
  /* the fixpoint is reached (no growth) well inside the iteration budget */
  EXPECT_LT(out["iterations"].get<int>(), 20);
  EXPECT_GE(out["solutionCount"].get<int>(), 0);
}

TEST(LogicEngine, Sample5_RejectsNonGroundFacts) {
  const json req = json{{"facts", json::array({"parent(X,bob)"})},
                        {"query", "parent(tom,bob)"}};
  const json out = phoenix::logic::runJson(req);
  EXPECT_FALSE(out["ok"].get<bool>());
  EXPECT_TRUE(out.contains("error"));
}

TEST(LogicEngine, Sample6_ObjectFormLiteralsWork) {
  const json req = json{
      {"facts",
       json::array({json{{"pred", "owns"},
                         {"args", json::array({"alice", "book1"})}}})},
      {"query",
       json{{"pred", "owns"},
            {"args", json::array({json{{"atom", "alice"}}, json{{"var", "what"}}})}}}};
  const json out = phoenix::logic::runJson(req);
  ASSERT_TRUE(out["ok"].get<bool>()) << out.dump();
  ASSERT_EQ(out["solutionCount"].get<int>(), 1);
  EXPECT_EQ(out["solutions"][0]["what"].get<std::string>(), "book1");
}

TEST(LogicEngine, Sample7_NumericArgumentsSurviveTheJsonFacade) {
  const json req = json{
      {"facts", json::array({json{{"pred", "age"},
                                  {"args", json::array({"carol", 41})}}})},
      {"rules",
       json::array({json{{"head", "over40(X)"},
                         {"body", json::array({json{{"pred", "age"},
                                                    {"args", json::array({"X", "A"})}},
                                               json{{"pred", ">"},
                                                    {"args", json::array({"A", 40})}}})}}})},
      {"query", "over40(Who)"}};
  const json out = phoenix::logic::runJson(req);
  ASSERT_TRUE(out["ok"].get<bool>()) << out.dump();
  ASSERT_EQ(out["solutionCount"].get<int>(), 1);
  EXPECT_EQ(out["solutions"][0]["Who"].get<std::string>(), "carol");
}
