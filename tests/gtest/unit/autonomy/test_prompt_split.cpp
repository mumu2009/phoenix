/* GTest unit tests for Phoenix v7.0 prompt split / prompt composer. */

#include "prompt_split.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>

using namespace phoenix::prompt;
using namespace phoenix::testing;

class PromptSplitTest : public HacktestBase {};

TEST_F(PromptSplitTest, SystemPromptDefault) {
    auto system = SystemPrompt::arthurDefault();
    /* Autonomy is one stream: no assistant identity / version banner. */
    EXPECT_TRUE(system.identity.empty());
    EXPECT_TRUE(system.version.empty());
    EXPECT_EQ(system.version.find("Helios"), std::string::npos);
    EXPECT_TRUE(system.constraints.empty());
    EXPECT_TRUE(system.coreDirective.empty());
}

TEST_F(PromptSplitTest, SystemPromptJsonRoundTrip) {
    SystemPrompt s;
    s.identity = "You are Phoenix.";
    s.version = "v7.0";
    s.constraints = "Be safe.";
    s.coreDirective = "Help the user.";

    auto j = s.toJson();
    auto loaded = SystemPrompt::fromJson(j);
    EXPECT_EQ(loaded.identity, s.identity);
    EXPECT_EQ(loaded.version, s.version);
    EXPECT_EQ(loaded.constraints, s.constraints);
    EXPECT_EQ(loaded.coreDirective, s.coreDirective);
}

TEST_F(PromptSplitTest, MemoryPromptJsonRoundTrip) {
    MemoryPrompt m;
    m.summary = "Summary";
    m.relevantFacts = {"a", "b"};
    m.activeGoals = {"g1"};
    m.emotionalTone = "optimistic";
    m.benefitHarmBias = "approach";

    auto j = m.toJson();
    auto loaded = MemoryPrompt::fromJson(j);
    EXPECT_EQ(loaded.summary, m.summary);
    EXPECT_EQ(loaded.relevantFacts.size(), 2u);
    EXPECT_EQ(loaded.activeGoals.size(), 1u);
    EXPECT_EQ(loaded.emotionalTone, m.emotionalTone);
    EXPECT_EQ(loaded.benefitHarmBias, m.benefitHarmBias);
}

TEST_F(PromptSplitTest, ComposeIncludesSystemAndMemory) {
    SystemPrompt system;
    system.identity = "You are Phoenix.";
    system.constraints = "Be safe.";
    system.coreDirective = "Help.";

    MemoryPrompt memory;
    memory.summary = "User likes cats.";
    memory.relevantFacts = {"Cat fact 1"};
    memory.activeGoals = {"Answer about cats"};
    memory.emotionalTone = "friendly";
    memory.benefitHarmBias = "approach";

    PromptComposer composer(system, memory);
    auto prompt = composer.compose("Tell me about cats.", true, "\n---\n");

    EXPECT_NE(prompt.find("You are Phoenix."), std::string::npos);
    EXPECT_NE(prompt.find("User likes cats."), std::string::npos);
    EXPECT_NE(prompt.find("Cat fact 1"), std::string::npos);
    EXPECT_NE(prompt.find("Answer about cats"), std::string::npos);
    EXPECT_NE(prompt.find("friendly"), std::string::npos);
    EXPECT_NE(prompt.find("approach"), std::string::npos);
    EXPECT_NE(prompt.find("Tell me about cats."), std::string::npos);
}

TEST_F(PromptSplitTest, ComposeCanOmitMemory) {
    SystemPrompt system;
    system.identity = "You are Phoenix.";

    MemoryPrompt memory;
    memory.summary = "Secret";

    PromptComposer composer(system, memory);
    auto prompt = composer.compose("Hello", false);
    EXPECT_EQ(prompt.find("Secret"), std::string::npos);
}

TEST_F(PromptSplitTest, ComposeMessagesStructure) {
    SystemPrompt system;
    system.identity = "You are Phoenix.";

    MemoryPrompt memory;
    memory.summary = "User likes dogs.";

    PromptComposer composer(system, memory);
    auto messages = composer.composeMessages("Hi", true);

    ASSERT_TRUE(messages.is_array());
    EXPECT_EQ(messages.size(), 1u);
    EXPECT_TRUE(messages[0].contains("content"));
    EXPECT_NE(messages[0]["content"].get<std::string>().find("Hi"),
              std::string::npos);
    EXPECT_NE(messages[0]["content"].get<std::string>().find("User likes dogs."),
              std::string::npos);
}

TEST_F(PromptSplitTest, FromContextBuildsMemory) {
    using namespace phoenix::context;
    std::vector<ContextEntry> context;
    ContextEntry e1;
    e1.role = "user";
    e1.content = "Hello";
    ContextEntry e2;
    e2.role = "assistant";
    e2.content = "Hi there";
    context.push_back(e1);
    context.push_back(e2);

    auto memory = PromptComposer::fromContext(context, 5);
    EXPECT_FALSE(memory.summary.empty());
    EXPECT_GE(memory.relevantFacts.size(), 1u);
    EXPECT_NE(memory.summary.find("2"), std::string::npos);
}
