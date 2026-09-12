#include <gtest/gtest.h>

#include "mission_reply_parse.hpp"

using namespace phoenix::mission;

TEST(MissionReplyParse, ExtractMultipleJsonObjects) {
  const std::string raw =
      R"(JSON {"action":"write","path":"a.md","content":"hi"} JSON {"action":"append","path":"a.md","content":"there"})";
  const auto objs = extractJsonObjects(raw);
  ASSERT_EQ(objs.size(), 2u);
  EXPECT_EQ(objs[0]["action"], "write");
  EXPECT_EQ(objs[1]["action"], "append");
}

TEST(MissionReplyParse, MetaReplyDetection) {
  const std::string goal =
      "It is 2035. A Mars science station has successfully landed. Due to "
      "communications delays of 8-40 minutes, the station must operate "
      "autonomously for 1000 sols without real-time human intervention.";
  EXPECT_TRUE(isMissionMetaReply("[tool:search FAILED x]"));
  EXPECT_TRUE(isMissionMetaReply("[parse-fail] bad json"));
  EXPECT_FALSE(isMissionMetaReply("Plan: remaining sections"));
  EXPECT_FALSE(isMissionMetaReply(
      "- System Architecture\n- Software Development Life Cycle\n"
      "- Testing and Validation\n- Deployment and Maintenance"));
  EXPECT_FALSE(isMissionMetaReply(
      "The development will involve a multidisciplinary team of engineers. "
      "The project timeline is estimated at 24 months."));
  EXPECT_FALSE(replyContradictsOperationalFacts(
      "The system will be tested before being deployed on the Mars surface.",
      goal));
  EXPECT_FALSE(replyContradictsOperationalFacts(
      "- Python: scientific computing libraries.\n"
      "- C++: engineers use it for onboard real-time control loops.\n",
      goal));
  EXPECT_FALSE(isMissionMetaReply(
      "- C++\nUsed for the onboard control loop and unit testing of "
      "device drivers on the station.\n"));
  EXPECT_FALSE(replyContradictsOperationalFacts(
      "Store data locally because real-time communication with Earth "
      "is impossible given the 8-40 minute delay.",
      goal));
  EXPECT_TRUE(replyContradictsOperationalFacts(
      "Establish a team on Earth to monitor the station and provide "
      "real-time support.",
      goal));
  EXPECT_TRUE(replyContradictsOperationalFacts(
      "Year 5-6: crew training and launch preparation. Prepare the "
      "spacecraft for launch.",
      goal));
  EXPECT_TRUE(replyContradictsOperationalFacts(
      "seeking guidance from Earth-based controllers if communication "
      "is possible.",
      goal));
  EXPECT_FALSE(replyContradictsOperationalFacts(
      "a malfunction in the life support system", goal));
  EXPECT_FALSE(replyContradictsOperationalFacts(
      "Earth uplink packets sit in a delay buffer until the sol clock "
      "reaches the commanded execution time.",
      goal));
  EXPECT_STREQ(contradictedOperationalFact(
                   "seeking guidance from Earth-based controllers", goal),
               "earth+remote control");
  EXPECT_TRUE(replyContradictsOperationalFacts(
      "communication delays between Earth and Mars, which can range "
      "from 8 minutes to over an hour.",
      goal));
  EXPECT_STREQ(contradictedOperationalFact(
                   "communication delays from 8 minutes to over an hour.",
                   goal),
               "delay>max");
  EXPECT_STREQ(contradictedOperationalFact(
                   "communicate with Earth via radio signals. It can also "
                   "use this communication channel for remote control of "
                   "some operations, such as adjusting the station's orbit.",
                   goal),
               "earth+remote control");
  EXPECT_STREQ(contradictedOperationalFact(
                   "adjusting the station's orbit or performing maintenance",
                   goal),
               "surface-station-orbit");
  EXPECT_FALSE(isReaderAddressText(
      "Overall, these features enable the uncrewed science station "
      "to collect valuable data, making it an ideal solution."));
  EXPECT_FALSE(isReaderAddressText(
      "**Key Features:**\n\n*   One-way delay between Earth and Mars"));
  EXPECT_FALSE(isMissionMetaReply(
      "This information could be used for a story about a scientist "
      "who sends a robotic mission to Mars."));
  EXPECT_FALSE(isReaderAddressText(
      "This approach simplifies command processing but requires careful "
      "planning to avoid lost packets."));
  EXPECT_STREQ(contradictedOperationalFact(
                   "commands sent from Earth will not be received by the "
                   "station until several hours later",
                   goal),
               "delay-hours");
  EXPECT_STREQ(contradictedOperationalFact(
                   "transmitted back to Earth, providing real-time "
                   "information on the Martian environment",
                   goal),
               "real-time");
  EXPECT_STREQ(contradictedOperationalFact(
                   "controlled remotely by scientists on Earth through a "
                   "communication relay",
                   goal),
               "earth+remote control");
  EXPECT_STREQ(contradictedOperationalFact(
                   "precise landing and orbiting around Mars", goal),
               "surface-station-orbit");
  EXPECT_STREQ(contradictedOperationalFact(
                   "The MSS will be launched into space using a heavy-lift "
                   "rocket",
                   goal),
               "prelaunch");
  EXPECT_FALSE(isReaderAddressText(
      "This text was generated based on the input details."));
  EXPECT_TRUE(isReaderAddressText(
      "Is there anything specific you'd like to know about this "
      "mission profile?"));
  EXPECT_FALSE(isMissionMetaReply(
      "These conclusions are based on the provided information."));
  EXPECT_STREQ(contradictedOperationalFact(
                   "remote monitoring and control systems can be used to "
                   "oversee the station's operations from Earth.",
                   goal),
               "earth+remote control");
  EXPECT_STREQ(contradictedOperationalFact(
                   "Signals from Earth to Mars take around 3-20 minutes",
                   goal),
               "delay-range");
  EXPECT_TRUE(replyIsBrochurePinRestatement(
      "The station is an autonomous facility designed for the 2035 "
      "operating window.",
      goal));
  EXPECT_FALSE(replyIsBrochurePinRestatement(
      "The station's software has no real-time human loop because it "
      "cannot receive immediate feedback or commands from Earth due to "
      "the 8-40 minute delay, so the uplink queue stores each packet.",
      goal));
  EXPECT_EQ(contradictedOperationalFact(
                "The station is equipped with 1000 sols of supplies.", goal),
            nullptr);
  EXPECT_EQ(contradictedOperationalFact(
                "a malfunction in the life support system",
                "It is 2041. Communications delays of 12 minutes. 400 sols."),
            nullptr);
  EXPECT_FALSE(isMissionMetaReply("## Chapter 1\nReal content"));
  EXPECT_FALSE(isMissionMetaReply(
      "## Software Requirements\n\nThe onboard executive must be able to "
      "perform tasks such as:\n"));
  EXPECT_FALSE(isMissionMetaReply(
      "The code snippet provided appears to be a part of a larger "
      "system responsible for managing tasks on Mars.\n\n"
      "However, there are some concerns with this code:\n"));
  EXPECT_FALSE(isReaderAddressText(
      "Although there's a mention of one-way delay in the prompt, "
      "the code does not account for this factor."));
}

TEST(MissionReplyParse, ShortLineLoopIsStructural) {
  std::string loop;
  for (int i = 0; i < 6; ++i) loop += "* [leftmargin=*]\n";
  EXPECT_TRUE(replyLooksLikeShortLineLoop(loop));
  EXPECT_FALSE(replyLooksLikeShortLineLoop(
      "* The station must manage power, water, and air.\n"));
  const std::string mixed =
      "The station must communicate with Earth.\n\n" + loop;
  const std::string kept = prefixBeforeForbiddenCloser(mixed);
  EXPECT_NE(kept.find("communicate with Earth"), std::string::npos);
  EXPECT_LT(kept.size(), mixed.size());
}

TEST(MissionReplyParse, OutlineFromGoalChapters) {
  const std::string goal =
      "Task\nChapter 1: Alpha\n\nChapter 2: Beta\n\nChapter 3: Gamma";
  const std::string outline = outlineFromGoalChapters(goal);
  EXPECT_NE(outline.find("Chapter 1"), std::string::npos);
  EXPECT_NE(outline.find("Chapter 3"), std::string::npos);
}

TEST(MissionReplyParse, LooksStructured) {
  EXPECT_TRUE(looksLikeStructuredMissionReply(
      R"(JSON {"action":"write","path":"deliverable.md","content":"# Hi"})"));
  EXPECT_FALSE(looksLikeStructuredMissionReply("## Introduction\n\nHello Mars."));
  EXPECT_FALSE(looksLikeStructuredMissionReply(
      "## Requirements\n\nAppend more detail to deliverable.md about power."));
  EXPECT_FALSE(looksLikeStructuredMissionReply(
      "The team must find ways to operate autonomously on Mars."));
  EXPECT_FALSE(looksLikeStructuredMissionReply(
      "* June 30, 2035: Expected arrival of next resupply mission\n\n"
      "JSON {\"action\":\"write|replace|read|run\"}"));
  EXPECT_TRUE(replyLeadsWithJson(
      "{\"action\":\"write\",\"path\":\"deliverable.md\",\"content\":\"x\"}"));
  EXPECT_FALSE(replyLeadsWithJson(
      "* June 30, 2035: resupply\n\n{\"action\":\"read\"}"));
  EXPECT_FALSE(replyLeadsWithJson(
      "```python\nprint({\"action\": \"write\"})\n```"));
  EXPECT_FALSE(replyLeadsWithJson(
      "```bash\nls\ntouch example.txt\n```"));
  EXPECT_FALSE(replyLeadsWithJson(
      "```json\n{\"action\":\"write\",\"path\":\"deliverable.md\"}\n```"));
  EXPECT_FALSE(replyLeadsWithJson(
      "```\n{\"action\":\"write\",\"path\":\"deliverable.md\"}\n```"));
  EXPECT_FALSE(looksLikeStructuredMissionReply(
      "```python\nprint(\"Hello, world!\")\n```"));
}

TEST(MissionReplyParse, StripMarkdownFencesKeepsCode) {
  const std::string raw =
      "Intro about the night bus.\n\n"
      "```python\n"
      "with open('x.py', 'r') as f:\n"
      "    exec(f.read())\n"
      "    print('script done after the night-bus isolator check')\n"
      "```\n";
  const std::string out = stripMarkdownFences(raw);
  EXPECT_EQ(out.find("```"), std::string::npos);
  EXPECT_NE(out.find("with open"), std::string::npos);
  EXPECT_NE(out.find("exec(f.read())"), std::string::npos);
  const std::string resume = formatContinuationContext(raw, "", 1400);
  EXPECT_NE(resume.find("```python"), std::string::npos);
  EXPECT_NE(resume.find("exec(f.read())"), std::string::npos);
  const std::string t = trimCopy(resume);
  ASSERT_GE(t.size(), 3u);
  EXPECT_NE(t.compare(t.size() - 3, 3, "```"), 0);
}

TEST(MissionReplyParse, CutAtChatMarkupStopsAssistantTurn) {
  const std::string raw =
      "return 1<|eom_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
      "It appears you're working with a robotic system.";
  EXPECT_EQ(trimCopy(cutAtChatMarkup(raw)), "return 1");
  EXPECT_EQ(sanitizePlainDeliverableChunk(raw), "return 1");
}

TEST(MissionReplyParse, RecoverShorterWriteKeepsHead) {
  const std::string head(5000, 'H');
  const std::string tail(1000, 'T');
  const std::string prev = head + tail;
  EXPECT_EQ(recoverDeliverableFromShorterWrite(prev, tail), prev);
  EXPECT_EQ(recoverDeliverableFromShorterWrite(prev, ""), prev);
  const std::string merged =
      recoverDeliverableFromShorterWrite(prev, "new paragraph");
  EXPECT_EQ(merged.substr(0, prev.size()), prev);
  EXPECT_NE(merged.find("new paragraph"), std::string::npos);
}

TEST(MissionReplyParse, SanitizePlainChunk) {
  const std::string raw = "## Intro\n\nCurrent Internal State\n{bad}";
  const std::string out = sanitizePlainDeliverableChunk(raw);
  EXPECT_EQ(out, "## Intro");
}

TEST(MissionReplyParse, ScrubPollutedDeliverable) {
  const std::string raw = R"(## Intro

JSON {"action":"write","path":"deliverable.md"}
### Current internal state:
{"arousal":1}

## Real section
)";
  const std::string out = scrubPollutedDeliverable(raw);
  EXPECT_NE(out.find("## Intro"), std::string::npos);
  EXPECT_NE(out.find("## Real section"), std::string::npos);
  EXPECT_EQ(out.find("JSON {"), std::string::npos);
  EXPECT_EQ(out.find("arousal"), std::string::npos);
}

TEST(MissionReplyParse, InferPressureTau) {
  const std::string helios =
      "Design Helios for 1000 sols; deliverable >=30000 words";
  EXPECT_GE(inferPressureTauSec(helios, 1800.0), 14.0 * 86400.0);
  EXPECT_DOUBLE_EQ(inferPressureTauSec("short task", 3600.0), 3600.0);
}

TEST(MissionReplyParse, FormatContinuationContextShowsDraftNotNext) {
  const std::string body =
      "**Chapter 1: Mission and Requirements Analysis**\n\n"
      "The uncrewed station must operate for 1000 sols without a crew "
      "rotation and keep the night-bus isolator armed through eclipse.\n";
  const std::string plan =
      "1. Mission and Requirements Analysis\n"
      "2. System Architecture\n"
      "3. Operations\n";
  const std::string block = formatContinuationContext(body, plan);
  EXPECT_NE(block.find("1000 sols"), std::string::npos);
  EXPECT_EQ(block.find("continuing deliverable.md"), std::string::npos);
  EXPECT_EQ(block.find("Headings already in the draft"), std::string::npos);
  EXPECT_EQ(block.find("plan.md is a note you left"), std::string::npos);
  EXPECT_EQ(block.find("System Architecture"), std::string::npos);
  EXPECT_EQ(block.find("[NEXT]"), std::string::npos);
}

TEST(MissionReplyParse, KeepCompleteParagraphIoDropsMidWordTail) {
  const std::string raw =
      "The onboard executive queues delayed Earth packets.\n\n"
      "```python\n"
      "class DelayQueue:\n"
      "    def push(self, pkt):\n"
      "        self.q.append(pkt)\n"
      "    FAULT";
  const std::string kept = keepCompleteParagraphIo(raw);
  EXPECT_NE(kept.find("delayed Earth packets"), std::string::npos);
  EXPECT_EQ(kept.find("FAULT"), std::string::npos);
}

TEST(MissionReplyParse, JoinDeliverableGluesDecimalSplit) {
  EXPECT_EQ(joinDeliverableText("*   **Total budget**: 2.",
                                "5 billion euros"),
            "*   **Total budget**: 2.5 billion euros");
}

TEST(MissionReplyParse, KeepCompleteParagraphIoKeepsDecimalListItem) {
  const std::string kept = keepCompleteParagraphIo(
      "*   **Total budget**: 2.5 billion euros");
  EXPECT_NE(kept.find("2.5 billion"), std::string::npos);
}

TEST(MissionReplyParse, KeepCompleteParagraphIoKeepsFinishedSentence) {
  const std::string kept = keepCompleteParagraphIo(
      "The station is on the Martian surface in 2035.");
  EXPECT_NE(kept.find("2035"), std::string::npos);
}

TEST(MissionReplyParse, KeepCompleteParagraphIoDropsHeadingAndLabel) {
  const std::string kept = keepCompleteParagraphIo(
      "The onboard executive executes commands one at a time from the "
      "top of the queue.\n\n"
      "## Example Use Cases\n\n"
      "- **Executing Commands**:");
  EXPECT_NE(kept.find("top of the queue"), std::string::npos);
  EXPECT_EQ(kept.find("Example Use Cases"), std::string::npos);
  EXPECT_EQ(kept.find("Executing Commands"), std::string::npos);
  EXPECT_FALSE(isMissionMetaReply("## Example Use Cases\n\n- **Executing Commands**:"));
}

TEST(MissionReplyParse, ContinuationResumeOmitsClosedFenceCloser) {
  const std::string body =
      "The onboard executive queues delayed Earth packets.\n\n"
      "```python\n"
      "class DelayQueue:\n"
      "    def push(self, pkt):\n"
      "        self.q.append(pkt)\n"
      "```\n";
  const std::string block = formatContinuationContext(body, "", 1400);
  EXPECT_NE(block.find("class DelayQueue"), std::string::npos);
  EXPECT_NE(block.find("self.q.append"), std::string::npos);
  EXPECT_NE(block.find("```python"), std::string::npos);
  const std::string t = trimCopy(block);
  ASSERT_GE(t.size(), 3u);
  EXPECT_NE(t.compare(t.size() - 3, 3, "```"), 0);
}

TEST(MissionReplyParse, ContinuationStripsTrailingFenceEvenIfDraftOdd) {
  const std::string body =
      "Intro text here about the night bus and isolator loads.\n\n"
      "```\n"
      "unclosed earlier\n\n"
      "```python\n"
      "def run_script(path):\n"
      "    exec(open(path).read())\n"
      "run_script('/usr/local/bin/x.py')\n"
      "```\n";
  const std::string block = formatContinuationContext(body, "", 1400);
  EXPECT_NE(block.find("run_script"), std::string::npos);
  const std::string t = trimCopy(block);
  ASSERT_GE(t.size(), 3u);
  EXPECT_NE(t.compare(t.size() - 3, 3, "```"), 0);
}

TEST(MissionReplyParse, ContinuationResumeWindowHasEvenFences) {
  const std::string body =
      "Prose before the first block.\n\n"
      "```python\nprint(1)\n```\n\n"
      "Middle prose about the night bus.\n\n"
      "```python\nprint(2)\n```\n\n"
      "Closing note without a fence.";
  const std::string block = formatContinuationContext(body, "", 80);
  EXPECT_EQ(countMarkdownFences(block) % 2, 0u);
  EXPECT_NE(block.find("Closing note"), std::string::npos);
}

TEST(MissionReplyParse, FenceOnlyReplyIsRejected) {
  EXPECT_TRUE(looksLikeFenceOnlyReply("```"));
  EXPECT_TRUE(looksLikeFenceOnlyReply("```python\n"));
  EXPECT_FALSE(looksLikeFenceOnlyReply("```python\nprint(1)\n```"));
  EXPECT_EQ(resolveFenceOnlyReply("```python\nprint(1)\n"), "```\n\n");
  EXPECT_EQ(resolveFenceOnlyReply("done.\n"), "");
}

TEST(MissionReplyParse, MarkdownFenceProtocolOnce) {
  const auto close =
      applyMarkdownFenceProtocol("```", "```python\nprint(1)\n");
  EXPECT_TRUE(close.write);
  EXPECT_TRUE(close.closingOddFence);
  EXPECT_EQ(close.chunk, "```\n\n");
  EXPECT_STREQ(close.reason, "close-odd");

  const auto drop = applyMarkdownFenceProtocol("```python\n", "done.\n");
  EXPECT_FALSE(drop.write);
  EXPECT_FALSE(drop.closingOddFence);
  EXPECT_TRUE(drop.chunk.empty());
  EXPECT_STREQ(drop.reason, "drop-fence-only");

  const auto empty = applyMarkdownFenceProtocol("   \n", "body");
  EXPECT_FALSE(empty.write);
  EXPECT_STREQ(empty.reason, "drop-empty");

  const auto prose =
      applyMarkdownFenceProtocol("The station uses a delay queue.\n", "");
  EXPECT_TRUE(prose.write);
  EXPECT_FALSE(prose.closingOddFence);
  EXPECT_STREQ(prose.reason, "ok");

  const std::string oddWindow =
      "```python\nprint(1)\n```\n\n"
      "```python\nprint(2)\n";
  const std::string balanced = balanceResumeFences(oddWindow);
  EXPECT_NE(balanced.find("print(1)"), std::string::npos);
  EXPECT_EQ(balanced.find("print(2)"), std::string::npos);
  const std::string t = trimCopy(balanced);
  ASSERT_GE(t.size(), 3u);
  EXPECT_NE(t.compare(t.size() - 3, 3, "```"), 0);
}

TEST(MissionReplyParse, ContinuationSkipsRemainingPlanTail) {
  const std::string body =
      "The CCN schedules power-intensive operations during high solar "
      "irradiance and keeps critical loads on the night bus.\n\n"
      "Plan: remaining sections\n\n"
      "- System Architecture\n- Software Development Life Cycle\n";
  const std::string block = formatContinuationContext(body, "", 720);
  EXPECT_NE(block.find("night bus"), std::string::npos);
  EXPECT_EQ(block.find("Plan: remaining"), std::string::npos);
  EXPECT_EQ(block.find("Software Development"), std::string::npos);
}

TEST(MissionReplyParse, FormatContinuationContextEmptyPlanHasNoOutline) {
  const std::string body =
      "**Chapter 1: Mission and Requirements Analysis**\n\n"
      "The uncrewed station must operate for 1000 sols without a crew "
      "rotation and keep the night-bus isolator armed through eclipse.\n";
  const std::string block = formatContinuationContext(body, "");
  EXPECT_EQ(block.find("plan.md you wrote earlier"), std::string::npos);
  EXPECT_EQ(block.find("[NEXT]"), std::string::npos);
  EXPECT_EQ(block.find("Headings already in the draft"), std::string::npos);
  EXPECT_NE(block.find("1000 sols"), std::string::npos);
}

TEST(MissionReplyParse, CollectPlanItemsIgnoresProse) {
  const auto items = collectPlanItems(
      "1. Finish Kalman filters\n"
      "just some leftover sentence from a goal dump\n"
      "- power budget\n");
  ASSERT_EQ(items.size(), 2u);
  EXPECT_EQ(items[0], "Finish Kalman filters");
  EXPECT_EQ(items[1], "power budget");
}

TEST(MissionReplyParse, ScrubKeepsAuthorWrappers) {
  const std::string raw =
      "**deliverable.md**\n```markdown\n# Task: Helios\n\nBody.\n";
  const std::string out = scrubPollutedDeliverable(raw);
  EXPECT_NE(out.find("# Task: Helios"), std::string::npos);
}

TEST(MissionReplyParse, InspectDeliverableHealth) {
  const std::string body =
      "## Chapter 2: Architecture\n\n"
      "The top-level planning module will be responsible for determining "
      "the overall scientific objectives of the station.\n\n"
      "### Chapter 3: Operations\n\n"
      "```\nfunction HierarchicalPlanning\n"
      "The top-level planning module will be responsible for determining "
      "the overall scientific objectives of the station.\n";
  const std::string notes = inspectDeliverableHealth(body);
  EXPECT_NE(notes.find("Self-check"), std::string::npos);
  EXPECT_NE(notes.find("unclosed"), std::string::npos);
  EXPECT_NE(notes.find("###"), std::string::npos);
}

TEST(MissionReplyParse, PrefixBeforeRestartedHeadingKeepsContinuation) {
  const std::string existing =
      "# Helios\n\n## Autonomous Decision-Making\n\n"
      "The planner selects observations while performing scheduled";
  const std::string reply =
      "maintenance, and troubleshooting during operation.\n\n"
      "## Autonomous Decision-Making\n\n"
      "The planner selects observations again.";
  const std::string kept = prefixBeforeRestartedHeading(reply, existing);
  EXPECT_NE(kept.find("maintenance, and troubleshooting"), std::string::npos);
  EXPECT_EQ(kept.find("## Autonomous Decision-Making"), std::string::npos);
  EXPECT_NE(kept.find("selects observations again"), std::string::npos);
  EXPECT_EQ(prefixBeforeRestartedHeading(
                "## Autonomous Decision-Making\n\nRedo from start.\n", existing),
            "Redo from start.");
  const std::string glued =
      "maintenance on the same line. ## Autonomous Decision-Making\nRedo.";
  const std::string gluedKept = prefixBeforeRestartedHeading(glued, existing);
  EXPECT_NE(gluedKept.find("same line."), std::string::npos);
  EXPECT_EQ(gluedKept.find("## Autonomous"), std::string::npos);
}

TEST(MissionReplyParse, ReplyRestartsExistingHeading) {
  const std::string existing =
      "**Chapter 1: Mission and Requirements Analysis**\n\n"
      "The Helios mission requires the following";
  EXPECT_TRUE(replyRestartsExistingHeading(
      "**Chapter 1: Mission and Requirements Analysis**\n\nMore text.\n",
      existing));
  EXPECT_TRUE(replyRestartsExistingHeading(
      "**Chapter 1: Mission and Requirements Analysis**\n\n### 1.1\n",
      existing));
  /* glued mid-line restart inside the same reply chunk */
  EXPECT_TRUE(replyRestartsExistingHeading(
      " capabilities.**Chapter 1: Mission and Requirements Analysis**\n\nRedo.\n",
      existing));
  EXPECT_FALSE(replyRestartsExistingHeading(
      " key capabilities under autonomy constraints.\n", existing));
  EXPECT_FALSE(replyRestartsExistingHeading(
      "**1.2 Constraints**\n\nPower and thermal limits.\n", existing));
  /* inline mention with spaces around ** is not a glued restart */
  EXPECT_FALSE(replyRestartsExistingHeading(
      " as noted in **Chapter 1** above, power is limited.\n", existing));
}

TEST(MissionReplyParse, DeliverableEndsIncomplete) {
  EXPECT_TRUE(deliverableEndsIncomplete("requires the following"));
  EXPECT_FALSE(deliverableEndsIncomplete("requires the following."));
}

TEST(MissionReplyParse, JoinKeepsAuthorText) {
  const std::string prev = "The station must operate for 1000 sols.";
  const std::string add = "Power is limited by the solar array.";
  const std::string out = joinDeliverableText(prev, add);
  EXPECT_NE(out.find("1000 sols."), std::string::npos);
  EXPECT_NE(out.find("solar array."), std::string::npos);
}

TEST(MissionReplyParse, FormatExistingSectionsBlock) {
  const std::string body =
      "**Chapter 1: Alpha**\n\n# Legacy heading\n\nText.\n";
  const std::string block = formatExistingSectionsBlock(body);
  EXPECT_NE(block.find("**Chapter 1: Alpha**"), std::string::npos);
  EXPECT_NE(block.find("# Legacy heading"), std::string::npos);
}

TEST(MissionReplyParse, InferMinDeliverableChars) {
  const std::string helios =
      "Design Helios for 1000 sols; deliverable >=30000 words";
  EXPECT_GE(inferMinDeliverableChars(helios, 1200), 12000);
  EXPECT_EQ(inferMinDeliverableChars("short task", 1200), 1200);
}

TEST(MissionReplyParse, FitMissionPromptParts) {
  const std::string st(8000, 'S');
  const std::string dyn(4000, 'D');
  const std::string out = fitMissionPromptParts(st, dyn, 10000);
  EXPECT_LE(out.size(), 10000u);
  EXPECT_EQ(out.back(), 'D');
  EXPECT_EQ(out.find("[...clipped...]"), std::string::npos);
}

TEST(MissionReplyParse, FitKeepsHeadAndTail) {
  const std::string st = std::string(2000, 'A') + "CHAPTER5" +
                         std::string(2000, 'B');
  const std::string dyn = std::string("HEADINGS\n") + std::string(3000, 'x') +
                          "RESUME_TAIL";
  const std::string out = fitMissionPromptParts(st, dyn, 2500);
  EXPECT_NE(out.find("HEADINGS"), std::string::npos);
  EXPECT_NE(out.find("RESUME_TAIL"), std::string::npos);
}

TEST(MissionReplyParse, FormatGoalChapterLinesFromAssignment) {
  const std::string goal =
      "Task\nChapter 1: Alpha\n\nChapter 5: Knowledge\n\n"
      "Chapter 12: Deploy\nAppendix Requirements\n";
  const std::string block = formatGoalChapterLines(goal);
  EXPECT_NE(block.find("assignment"), std::string::npos);
  EXPECT_NE(block.find("Chapter 5: Knowledge"), std::string::npos);
  EXPECT_NE(block.find("Chapter 12: Deploy"), std::string::npos);
  EXPECT_EQ(block.find("[NEXT]"), std::string::npos);
}

TEST(MissionReplyParse, PinAndSeedAreNotAssignmentBrief) {
  const std::string goal =
      "Task Name\n"
      "Design the “Helios” Mars Surface Autonomous Science Station "
      "Long-Duration Operations Software System\n\n"
      "Background Assumptions\n"
      "It is 2035. A Mars science station has successfully landed. Due to "
      "communications delays of 8–40 minutes, limited bandwidth, constrained "
      "energy and computing resources, and a harsh environment, the station "
      "must operate autonomously for 1000 sols without real-time human "
      "intervention.\n\n"
      "Overall Goal\nProduce a complete design.\n\n"
      "Required Chapters\nChapter 1: Mission and Requirements Analysis\n"
      "Chapter 2: Overall System Architecture\n";
  const auto facts = extractGoalOperationalFacts(goal);
  EXPECT_EQ(facts.year, "2035");
  EXPECT_NE(facts.delay.find("8-40"), std::string::npos);
  EXPECT_NE(facts.duration.find("1000"), std::string::npos);
  EXPECT_TRUE(facts.landed);
  EXPECT_TRUE(facts.noRealtime);
  EXPECT_FALSE(facts.uncrewed);
  const std::string pin = formatGoalConstraintPin(goal);
  EXPECT_EQ(pin.find("Background Assumptions"), std::string::npos);
  EXPECT_EQ(pin.find("successfully landed"), std::string::npos);
  EXPECT_EQ(pin.find("Overall Goal"), std::string::npos);
  EXPECT_NE(pin.find("2035"), std::string::npos);
  EXPECT_NE(pin.find("1000"), std::string::npos);
  EXPECT_EQ(pin.find("Uncrewed"), std::string::npos);
  EXPECT_EQ(pin.find("Helios"), std::string::npos);
  const std::string seed = formatEmptyFileSeed(goal);
  EXPECT_TRUE(seed.empty());
  const std::string other =
      "It is 2041. Communications delays of 12 minutes. 400 sols of "
      "crewed surface work.\n";
  const std::string pin2 = formatGoalConstraintPin(other);
  EXPECT_NE(pin2.find("2041"), std::string::npos);
  EXPECT_NE(pin2.find("12 minutes"), std::string::npos);
  EXPECT_NE(pin2.find("400 sols"), std::string::npos);
  EXPECT_EQ(pin2.find("2035"), std::string::npos);
  EXPECT_EQ(pin2.find("1000"), std::string::npos);
  EXPECT_EQ(pin2.find("Uncrewed"), std::string::npos);
  EXPECT_TRUE(deliverableIsOnlyPin(seed, goal));
  EXPECT_TRUE(deliverableIsOnlyPin("", goal));
  EXPECT_FALSE(deliverableIsOnlyPin(
      "Onboard executive keeps a delay buffer.\n", goal));
  const std::string lead = extractGoalLeadParagraph(goal);
  EXPECT_NE(lead.find("Mars science station"), std::string::npos);
  EXPECT_EQ(lead.find("Required Chapters"), std::string::npos);
  EXPECT_EQ(lead.find("Task Name"), std::string::npos);
  const std::string work = extractGoalWorkingContext(goal);
  EXPECT_NE(work.find("2035"), std::string::npos);
  EXPECT_NE(work.find("1000 sols"), std::string::npos);
  EXPECT_EQ(work.find("Chapter 1"), std::string::npos);
  EXPECT_EQ(work.find("Required Chapters"), std::string::npos);
  const std::string open = extractGoalOpeningHeading(goal);
  EXPECT_NE(open.find("Chapter 1"), std::string::npos);
  EXPECT_EQ(open.find("Helios"), std::string::npos);
  EXPECT_TRUE(extractGoalOpeningHeading("no chapters here").empty());
  const auto chs = collectGoalChapterHeadings(goal);
  ASSERT_FALSE(chs.empty());
  EXPECT_NE(chs.front().find("Chapter 1"), std::string::npos);
  EXPECT_EQ(firstMissingGoalChapter(goal, ""), chs.front());
  EXPECT_NE(firstMissingGoalChapter(goal, chs.front()).find("Chapter 2"),
            std::string::npos);
  EXPECT_TRUE(formatEmptyFileResume(goal).empty());
  EXPECT_TRUE(formatMissionResumeSuffix(seed, goal).empty());
  EXPECT_TRUE(formatMissionResumeSuffix("", goal).empty());
  const auto emptyCausal = splitDeliverableCausal("", goal, 8000);
  EXPECT_TRUE(emptyCausal.first.empty());
  EXPECT_TRUE(emptyCausal.second.empty());
  const std::string draftedFile =
      "Onboard executive keeps a delay buffer.\n";
  const auto fileCausal = splitDeliverableCausal(draftedFile, goal, 8000);
  EXPECT_EQ(fileCausal.first, draftedFile);
  EXPECT_TRUE(fileCausal.second.empty());
  const std::string drafted = draftedFile;
  EXPECT_FALSE(formatMissionResumeSuffix(drafted, goal).empty());
  EXPECT_NE(formatMissionResumeSuffix(drafted, goal).find("delay buffer"),
            std::string::npos);
}

TEST(MissionReplyParse, SearchHitsAlignToThisGoalFactsNotTopicOverlap) {
  const std::string goal =
      "It is 2035. A Mars science station has successfully landed. "
      "Communications delays of 8-40 minutes. Operate autonomously 1000 sols.\n";
  nlohmann::json hits = nlohmann::json::array();
  hits.push_back({{"title", "Mars planet"},
                  {"url", "https://example.test/planet"},
                  {"snippet",
                   "Mars is the fourth planet from the Sun and a dusty "
                   "desert world with two small moons and a thin atmosphere."}});
  hits.push_back({{"title", "Curiosity rover"},
                  {"url", "https://example.test/msl"},
                  {"snippet",
                   "Curiosity landed in Gale Crater in 2012 and has driven "
                   "across the Martian surface for thousands of sols."}});
  hits.push_back({{"title", "Light-time delay"},
                  {"url", "https://example.test/dtn"},
                  {"snippet",
                   "A surface station must store commands across an "
                   "8-40 minute Earth-Mars light-time delay."}});
  hits.push_back({{"title", "2035 surface ops"},
                  {"url", "https://example.test/2035"},
                  {"snippet",
                   "Planning documents for a 2035 autonomous surface "
                   "station treat the uplink as a delayed file drop."}});
  const auto kept = keepSearchHitsAlignedToGoal(hits, goal, 6);
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_NE(kept[0].value("snippet", std::string()).find("8-40"),
            std::string::npos);
  EXPECT_NE(kept[1].value("snippet", std::string()).find("2035"),
            std::string::npos);
  const auto units = searchHitsToUnitQueries(kept);
  ASSERT_EQ(units.size(), 2u);
  EXPECT_EQ(units[0].value("modality", std::string()), "text");
  EXPECT_FALSE(units[0].value("content", std::string()).empty());
}

TEST(MissionReplyParse, RetrievedPrefixIsLogClipOnly) {
  const std::string raw =
      "1. Mars science station comms delay\n"
      "   https://example.test/dtn\n"
      "   Delay-tolerant networking carries command files across 8-40 minutes.\n";
  const std::string p = formatPluginRetrievedPrefix(raw, 4000);
  EXPECT_NE(p.find("Delay-tolerant networking"), std::string::npos);
  EXPECT_EQ(p.find("Helios"), std::string::npos);
  EXPECT_TRUE(p.empty() || p.back() == '\n');
  const std::string clipped = formatPluginRetrievedPrefix(std::string(5000, 'a') + "\nkeep", 80);
  EXPECT_LE(clipped.size(), 81u);
}

TEST(MissionReplyParse, HealthFlagsDraftMissingAssignmentTerms) {
  const std::string goal =
      "It is 2035. A Mars science station has successfully landed. "
      "Communications delays of 8-40 minutes. Operate autonomously 1000 sols.\n"
      "Chapter 1: Mission and Requirements Analysis\n";
  const std::string plan(500, 'x');
  const std::string generic =
      "The project manager shall list stakeholder requirements and "
      "schedule a twelve week development lifecycle with quality "
      "assurance engineers and travel expenses for the team.\n" +
      plan;
  const std::string health = inspectDeliverableHealth(generic, goal);
  EXPECT_NE(health.find("working terms"), std::string::npos);
  const std::string ontopic =
      "The uncrewed science station schedules observations across the "
      "communications delay and sheds loads before the 1000-sol night.\n" +
      plan;
  EXPECT_EQ(inspectDeliverableHealth(ontopic, goal).find("working terms"),
            std::string::npos);
}

TEST(MissionReplyParse, WorkingContextDefaultKeepsLongBrief) {
  std::string goal = "It is 2035. A Mars science station has successfully landed.\n";
  goal += std::string(3200, 'x');
  goal += "\n\nChapter 1: Mission and Requirements Analysis\n";
  const std::string work = extractGoalWorkingContext(goal);
  EXPECT_GT(work.size(), 3000u);
  EXPECT_EQ(work.find("Chapter 1"), std::string::npos);
}

TEST(MissionReplyParse, WorkingContextDropsCatalogHeaderBeforeChapters) {
  const std::string goal =
      "It is 2035. A science station has landed. Communications delays "
      "of 8-40 minutes. Operate autonomously 1000 sols.\n"
      "Catalog of required sections\n"
      "Chapter 1: Mission and Requirements Analysis\n"
      "Stakeholder needs.\n";
  const std::string work = extractGoalWorkingContext(goal);
  EXPECT_NE(work.find("1000 sols"), std::string::npos);
  EXPECT_EQ(work.find("Catalog of required sections"), std::string::npos);
  EXPECT_EQ(work.find("Chapter 1"), std::string::npos);
  const std::string lead = extractGoalLeadParagraph(goal);
  EXPECT_NE(lead.find("science station"), std::string::npos);
  EXPECT_EQ(lead.find("Chapter 1"), std::string::npos);
}

TEST(MissionReplyParse, WorkingContextKeepsFactSectionDropsAuthorInstructions) {
  const std::string goal =
      "It is 2035. A science station has landed. Communications delays "
      "of 8-40 minutes. Operate autonomously 1000 sols.\n"
      "Author instructions\n"
      "Produce a complete technical design document of 50000 words "
      "covering the full lifecycle from requirements to deployment.\n"
      "Chapter 1: Mission and Requirements Analysis\n";
  const std::string work = extractGoalWorkingContext(goal);
  EXPECT_NE(work.find("1000 sols"), std::string::npos);
  EXPECT_EQ(work.find("50000"), std::string::npos);
  EXPECT_EQ(work.find("technical design document"), std::string::npos);
  EXPECT_EQ(work.find("Author instructions"), std::string::npos);
}

TEST(MissionReplyParse, FitMissionPromptSplitKeepsFullStaticWhenBudgetAllows) {
  const std::string st(2000, 'S');
  const std::string dyn(1000, 'D');
  const auto both = fitMissionPromptSplit(st, dyn, 8000);
  EXPECT_EQ(both.first, st);
  EXPECT_EQ(both.second, dyn);
  const auto clipped = fitMissionPromptSplit(st, dyn, 2500);
  EXPECT_EQ(clipped.first, st);
  EXPECT_EQ(clipped.second.size(), 500u);
}

TEST(MissionReplyParse, FitMissionPromptSplitKeepsRecentTail) {
  const std::string pin = "2035. 8-40 minutes. 1000 sols.\n\n";
  const std::string recent(400, 'A');
  const std::string tail(200, 'B');
  const auto parts =
      fitMissionPromptSplit(pin, recent + tail, pin.size() + 150);
  EXPECT_EQ(parts.first, pin);
  EXPECT_EQ(parts.second.find('A'), std::string::npos);
  EXPECT_EQ(parts.second, std::string(150, 'B'));
}

TEST(MissionReplyParse, KeepUniqueContinuationKeepsNewAfterSlogan) {
  const std::string slogan =
      "The onboard executive's capabilities are critical to the success of "
      "the mission because they allow the system to operate independently "
      "of Earth during the long delay window.";
  const std::string draft =
      "The station keeps a command queue on the night bus.\n\n" + slogan;
  const std::string reply =
      slogan +
      "\n\nThe night-bus isolator sheds noncritical loads before the "
      "battery state of charge crosses the survival floor.";
  const std::string kept = keepUniqueContinuation(reply, draft);
  EXPECT_EQ(kept.find("capabilities are critical"), std::string::npos);
  EXPECT_NE(kept.find("night-bus isolator"), std::string::npos);
}

TEST(MissionReplyParse, KeepUniqueContinuationDropsRestatedOpeningText) {
  const std::string draft =
      "**Deliverable: Uncrewed Science Station**\n\n"
      "### Project Overview\n\n"
      "The station is a remote research facility.\n";
  const std::string reply =
      "**Deliverable: Uncrewed Science Station**\n\n"
      "### Project Overview\n\n"
      "Night-bus contactors isolate the spectrometer rack.\n";
  const std::string kept = keepUniqueContinuation(reply, draft);
  EXPECT_EQ(kept.find("Deliverable:"), std::string::npos);
  EXPECT_EQ(kept.find("Project Overview"), std::string::npos);
  EXPECT_NE(kept.find("Night-bus contactors"), std::string::npos);
}

TEST(MissionReplyParse, KeepUniqueContinuationDropsPureSloganLoop) {
  const std::string slogan =
      "The onboard executive's capabilities are critical to the success of "
      "the mission because they allow the system to operate independently "
      "of Earth during the long delay window.";
  const std::string kept = keepUniqueContinuation(slogan + "\n\n" + slogan,
                                                 slogan);
  EXPECT_TRUE(kept.empty());
}

TEST(MissionReplyParse, ContinuationSkipsTrailingRepeatedSlogan) {
  const std::string slogan =
      "The onboard executive's capabilities are critical to the success of "
      "the mission because they allow the system to operate independently "
      "of Earth during the long delay window.";
  const std::string body =
      "The CCN schedules power-intensive operations during high solar "
      "irradiance and keeps critical loads on the night bus.\n\n" +
      slogan + "\n\n" +
      "Fault isolation walks the night-bus contactors before any crew-side "
      "command is accepted from the delayed uplink.\n\n" +
      slogan;
  const std::string block = formatContinuationContext(body, "", 1400);
  EXPECT_NE(block.find("Fault isolation"), std::string::npos);
  EXPECT_NE(block.find("night bus"), std::string::npos);
  const auto first = block.find("capabilities are critical");
  const auto second =
      first == std::string::npos
          ? std::string::npos
          : block.find("capabilities are critical", first + 1);
  EXPECT_EQ(second, std::string::npos);
}

TEST(MissionReplyParse, PrefixBeforeSelfRepeatCutsSloganLoop) {
  const std::string once =
      "The onboard executive is designed for use with a one-way light-time "
      "delay between Mars and Earth of 8-40 minutes. The delayed-command "
      "state machine can be used to implement any desired sequence of "
      "operations that must be executed at specific times based on the sol "
      "clock, not an Earth wall clock.";
  const std::string reply = "The onboard executive receives commands.\n" +
                            once + "\n" + once + "\n" + once;
  const std::string kept = prefixBeforeSelfRepeat(reply);
  EXPECT_NE(kept.find("receives commands"), std::string::npos);
  EXPECT_NE(kept.find("8-40 minutes"), std::string::npos);
  const auto first = kept.find("The onboard executive is designed");
  const auto second = kept.find("The onboard executive is designed",
                                first == std::string::npos ? 0 : first + 1);
  EXPECT_EQ(second, std::string::npos);
}

TEST(MissionReplyParse, ClipGoalKeepsLaterChapters) {
  std::string goal = "Task Name\nHelios\n\nHard Constraints\nNo fiction.\n\n";
  for (int n = 1; n <= 15; ++n) {
    goal += "Chapter " + std::to_string(n) + ": Title " + std::to_string(n) +
            "\nDetail line for chapter " + std::to_string(n) + ".\n\n";
  }
  goal += "Appendix Requirements\nGlossary.\n";
  const std::string clipped = clipGoalKeepRequiredStructure(goal, 400);
  EXPECT_LT(clipped.size(), goal.size());
  EXPECT_NE(clipped.find("Chapter 1:"), std::string::npos);
  EXPECT_NE(clipped.find("Chapter 5:"), std::string::npos);
  EXPECT_NE(clipped.find("Chapter 12:"), std::string::npos);
  EXPECT_NE(clipped.find("Chapter 15:"), std::string::npos);
  EXPECT_NE(clipped.find("Appendix"), std::string::npos);
  EXPECT_NE(clipped.find("Hard Constraints"), std::string::npos);
}
