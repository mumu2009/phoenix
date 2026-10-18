#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "addons/ThePlugInForSecurity/security_core.hpp"
#include "emergency_stop.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#endif

namespace fs = std::filesystem;
using phoenix::secamp::allowArtifactLoad;
using phoenix::secamp::auditPeFile;
using phoenix::secamp::auditWorkspaceRoot;
using phoenix::secamp::bindEstopProbes;
using phoenix::secamp::confineToWorkspace;
using phoenix::secamp::ensureProcessHardEmbed;
using phoenix::secamp::estopRequiresBlock;
using phoenix::secamp::mitigationsSealed;
using phoenix::secamp::noteWorkspaceRoot;
using phoenix::secamp::notifyMemeBarrierAttached;
using phoenix::secamp::safetyWiringClosed;
using phoenix::secamp::sha256Bytes;
using phoenix::secamp::sha256File;
using phoenix::secamp::testOnlyForceUnsealed;
using phoenix::secamp::testOnlyResetWiring;
using phoenix::secamp::testOnlySetHashCap;
using phoenix::secamp::verifyPinnedFile;
using phoenix::secamp::SecurityObservatory;

namespace {

void clearEnv(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

void bindRealEstop() {
  bindEstopProbes(
      []() { return phoenix::safety::EmergencyStop::instance().latched(); },
      []() { return phoenix::safety::EmergencyStop::instance().hasShutdownHandler(); });
  notifyMemeBarrierAttached();
}

class HardEmbedTest : public ::testing::Test {
protected:
  fs::path root;

  void SetUp() override {
    testOnlyResetWiring();
    phoenix::safety::EmergencyStop::instance().setShutdownHandler({});
    phoenix::safety::EmergencyStop::instance().resetForTesting();
    const auto report = ensureProcessHardEmbed();
    ASSERT_TRUE(report.sealed) << report.detail;
    bindRealEstop();
    root = fs::temp_directory_path() / "phoenix_sec_ws";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
  }

  void TearDown() override {
    phoenix::safety::EmergencyStop::instance().setShutdownHandler({});
    phoenix::safety::EmergencyStop::instance().resetForTesting();
    testOnlyResetWiring();
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};

} // namespace

TEST(HardEmbedHash, KnownVectors) {
  EXPECT_EQ(sha256Bytes("abc", 3),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256Bytes("", 0),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_F(HardEmbedTest, ProcessMitigationsReadBack) {
  const auto r = ensureProcessHardEmbed();
  EXPECT_TRUE(r.depImage);
  EXPECT_TRUE(r.aslrImage);
  EXPECT_TRUE(r.highEntropyImage);
  EXPECT_TRUE(r.depPolicy);
  EXPECT_TRUE(r.extensionPointsDisabled);
  EXPECT_TRUE(r.noRemoteImages);
  EXPECT_TRUE(r.noLowMandatoryLabelImages);
  EXPECT_TRUE(r.preferSystem32Images);
  EXPECT_TRUE(r.heapTerminateOnCorruption);
  EXPECT_TRUE(mitigationsSealed());
  EXPECT_EQ(r.detail, "ok");
}

TEST(HardEmbedImage, HostExecutableHasDepAslr) {
  const auto pe = auditPeFile("phoenix_main.exe");
  ASSERT_TRUE(pe.ok) << pe.detail;
  EXPECT_TRUE(pe.dep);
  EXPECT_TRUE(pe.aslr);
  EXPECT_TRUE(pe.highEntropy);
}

TEST_F(HardEmbedTest, UnsealedFailsClosed) {
  testOnlyForceUnsealed(true);
  EXPECT_FALSE(mitigationsSealed());
  const auto path = confineToWorkspace(root.string(), "a.md");
  EXPECT_FALSE(path.allowed);
  EXPECT_EQ(path.reason, "mitigations-not-sealed");
  const auto load = allowArtifactLoad(root.string() + "/a.md", std::string(64, 'a'));
  EXPECT_FALSE(load.allowed);
  EXPECT_EQ(load.reason, "mitigations-not-sealed");
}

TEST_F(HardEmbedTest, UnwiredFailsClosed) {
  testOnlyResetWiring();
  (void)ensureProcessHardEmbed();
  const auto load = allowArtifactLoad((root / "a.md").string(), std::string(64, 'a'));
  EXPECT_FALSE(load.allowed);
  EXPECT_EQ(load.reason, "safety-unwired");
  EXPECT_FALSE(safetyWiringClosed());
  EXPECT_FALSE(estopRequiresBlock());
}

TEST_F(HardEmbedTest, WorkspaceEscapeAndDevicesDenied) {
  EXPECT_FALSE(confineToWorkspace(root.string(), "../outside.txt").allowed);
  EXPECT_EQ(confineToWorkspace(root.string(), "../outside.txt").reason, "path-escape");
  EXPECT_EQ(confineToWorkspace(root.string(), "a/../../outside.txt").reason, "path-escape");
  EXPECT_EQ(confineToWorkspace(root.string(), "note.txt:stream").reason, "alternate-stream");
  EXPECT_EQ(confineToWorkspace(root.string(), "NUL").reason, "device-name");
  EXPECT_EQ(confineToWorkspace(root.string(), "dir/COM1").reason, "device-name");
  EXPECT_EQ(confineToWorkspace(root.string(), "C:/Windows/notepad.exe").reason, "absolute-path");
  EXPECT_EQ(confineToWorkspace(root.string(), "").reason, "empty-path");
  const auto ok = confineToWorkspace(root.string(), "notes/a.md");
  EXPECT_TRUE(ok.allowed) << ok.reason;
  EXPECT_EQ(ok.reason, "ok");
}

TEST_F(HardEmbedTest, SensitiveRootDenied) {
#ifdef _WIN32
  char buf[MAX_PATH];
  ASSERT_GT(GetWindowsDirectoryA(buf, MAX_PATH), 0u);
  const auto denied = confineToWorkspace(buf, "Temp/x.txt");
  EXPECT_FALSE(denied.allowed);
  EXPECT_EQ(denied.reason, "sensitive-path");
  const auto audit = auditWorkspaceRoot(buf);
  EXPECT_FALSE(audit.allowed);
  EXPECT_EQ(audit.reason, "sensitive-path");
#else
  const auto denied = confineToWorkspace("/etc", "passwd");
  EXPECT_FALSE(denied.allowed);
  EXPECT_EQ(denied.reason, "sensitive-path");
#endif
  const auto rel = auditWorkspaceRoot("./runtime_store/mission_workspace");
  EXPECT_TRUE(rel.allowed) << rel.reason;
  noteWorkspaceRoot("./runtime_store/mission_workspace");
}

#ifdef _WIN32
bool makeDirectoryJunction(const fs::path &link, const fs::path &target) {
  std::error_code ec;
  fs::remove_all(link, ec);
  if (!CreateDirectoryW(link.wstring().c_str(), nullptr))
    return false;
  std::wstring dest = L"\\??\\" + target.wstring();
  if (dest.empty() || dest.back() != L'\\')
    dest.push_back(L'\\');
  const std::wstring print = target.wstring();
  const USHORT substBytes = static_cast<USHORT>(dest.size() * sizeof(wchar_t));
  const USHORT printBytes = static_cast<USHORT>(print.size() * sizeof(wchar_t));
  const USHORT dataBytes =
      static_cast<USHORT>(8 + substBytes + sizeof(wchar_t) + printBytes + sizeof(wchar_t));
  std::vector<unsigned char> raw(8 + dataBytes, 0);
  struct Head {
    ULONG tag;
    USHORT len;
    USHORT reserved;
    USHORT substOff;
    USHORT substLen;
    USHORT printOff;
    USHORT printLen;
    wchar_t buf[1];
  };
  auto *head = reinterpret_cast<Head *>(raw.data());
  head->tag = IO_REPARSE_TAG_MOUNT_POINT;
  head->len = dataBytes;
  head->substOff = 0;
  head->substLen = substBytes;
  head->printOff = static_cast<USHORT>(substBytes + sizeof(wchar_t));
  head->printLen = printBytes;
  std::memcpy(head->buf, dest.c_str(), substBytes);
  std::memcpy(reinterpret_cast<unsigned char *>(head->buf) + head->printOff, print.c_str(),
              printBytes);
  HANDLE h = CreateFileW(link.wstring().c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    return false;
  DWORD returned = 0;
  const BOOL ok = DeviceIoControl(h, FSCTL_SET_REPARSE_POINT, raw.data(),
                                  static_cast<DWORD>(8 + dataBytes), nullptr, 0, &returned,
                                  nullptr);
  CloseHandle(h);
  return ok == TRUE;
}
#endif

TEST_F(HardEmbedTest, ReparseEscapeDenied) {
#ifdef _WIN32
  const fs::path outside = fs::temp_directory_path() / "phoenix_sec_out";
  std::error_code ec;
  fs::remove_all(outside, ec);
  ASSERT_TRUE(fs::create_directories(outside));
  const fs::path link = root / "jump";
  fs::remove_all(link, ec);
  const DWORD symFlags = SYMBOLIC_LINK_FLAG_DIRECTORY | 0x2;
  const bool linked =
      CreateSymbolicLinkW(link.wstring().c_str(), outside.wstring().c_str(), symFlags) == TRUE ||
      makeDirectoryJunction(link, outside);
  ASSERT_TRUE(linked) << "reparse point was not created, GetLastError=" << GetLastError();
  const auto d = confineToWorkspace(root.string(), "jump/secret.txt");
  EXPECT_FALSE(d.allowed) << d.canonical;
  EXPECT_EQ(d.reason, "path-escape");
  fs::remove_all(link, ec);
  fs::remove_all(outside, ec);
#else
  GTEST_SKIP() << "windows reparse point";
#endif
}

TEST_F(HardEmbedTest, PinMatchTamperAndFailClosed) {
  const fs::path file = root / "plugin.json";
  {
    std::ofstream out(file, std::ios::binary);
    out << "plugin-body";
  }
  std::string err;
  const std::string pin = sha256File(file.string(), &err);
  ASSERT_TRUE(err.empty()) << err;
  ASSERT_EQ(pin.size(), 64u);

  const auto upper = [&] {
    std::string u = pin;
    for (char &c : u)
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return u;
  }();
  const auto ok = allowArtifactLoad(file.string(), upper);
  EXPECT_TRUE(ok.allowed) << ok.reason;
  EXPECT_EQ(ok.sha256, pin);

  {
    std::ofstream out(file, std::ios::binary | std::ios::app);
    out << "x";
  }
  const auto bad = allowArtifactLoad(file.string(), pin);
  EXPECT_FALSE(bad.allowed);
  EXPECT_EQ(bad.reason, "mismatch");

  const auto bare = verifyPinnedFile(file.string(), "");
  EXPECT_FALSE(bare.allowed);
  EXPECT_EQ(bare.reason, "unpinned");
  EXPECT_EQ(allowArtifactLoad(file.string(), "abcd").reason, "bad-pin");
  EXPECT_EQ(allowArtifactLoad((root / "missing.json").string(), pin).reason, "missing");
  EXPECT_EQ(allowArtifactLoad(root.string() + "/../x.json", pin).reason, "path-escape");

#ifdef _WIN32
  const auto sys = allowArtifactLoad("C:\\Windows\\System32\\kernel32.dll", pin);
  EXPECT_FALSE(sys.allowed);
  EXPECT_EQ(sys.reason, "sensitive-path");
#endif

  testOnlySetHashCap(4);
  const auto huge = allowArtifactLoad(file.string(), pin);
  EXPECT_FALSE(huge.allowed);
  EXPECT_EQ(huge.reason, "too-large");
}

TEST_F(HardEmbedTest, EstopLatchFailsClosedAndStaysLatched) {
  int hits = 0;
  phoenix::safety::EmergencyStop::instance().setShutdownHandler([&] { ++hits; });
  EXPECT_TRUE(safetyWiringClosed());
  const auto first = phoenix::safety::EmergencyStop::instance().press("security-line");
  EXPECT_TRUE(first.value("latched", false));
  EXPECT_FALSE(first.value("alreadyLatched", true));
  EXPECT_EQ(hits, 1);
  const auto second = phoenix::safety::EmergencyStop::instance().press("again");
  EXPECT_TRUE(second.value("alreadyLatched", false));
  EXPECT_EQ(hits, 1);
  EXPECT_TRUE(estopRequiresBlock());
  EXPECT_FALSE(safetyWiringClosed());

  const fs::path file = root / "cfg.json";
  {
    std::ofstream out(file, std::ios::binary);
    out << "{}";
  }
  std::string err;
  const std::string pin = sha256File(file.string(), &err);
  ASSERT_TRUE(err.empty());
  const auto gate = allowArtifactLoad(file.string(), pin);
  EXPECT_FALSE(gate.allowed);
  EXPECT_EQ(gate.reason, "emergency-stop");
  const auto boxed = confineToWorkspace(root.string(), "a.md");
  EXPECT_FALSE(boxed.allowed);
  EXPECT_EQ(boxed.reason, "emergency-stop");
}

TEST(HardEmbedSoftBarrier, AnomalousMemeStillBlocksAndCleanDoesNot) {
  clearEnv("PHOENIX_SECURITY_ENABLED");
  clearEnv("PHOENIX_SECURITY_DEFENSE_OFF");
  clearEnv("PHOENIX_SECURITY_CONFIG_PIN");
  SecurityObservatory::instance().resetForTests();
  phoenix::secamp::DiscreteGraph g;
  g.ids = {"hub", "p1", "p2", "p3"};
  g.layers = {"meme", "meme", "meme", "meme"};
  g.mapped = {{"hubword"}, {"w1"}, {"w2"}, {"w3"}};
  g.edges = {{0, 1, 1.0}, {0, 2, 1.0}, {0, 3, 1.0}};
  g.anomalous.insert("hub");
  SecurityObservatory::instance().ingest(g);
  EXPECT_FALSE(SecurityObservatory::instance().pluginEnabled());
  auto open = SecurityObservatory::instance().inspectText("please recall hubword now");
  EXPECT_FALSE(open.blocked);
  ASSERT_TRUE(SecurityObservatory::instance().setPluginEnabled(true));
  auto blocked = SecurityObservatory::instance().inspectText("please recall hubword now");
  EXPECT_TRUE(blocked.blocked);
  EXPECT_EQ(blocked.reason, "high-impact-anomalous-meme");
  auto clean = SecurityObservatory::instance().inspectText("please recall ordinary notes");
  EXPECT_FALSE(clean.blocked);
  const auto defense = SecurityObservatory::instance().defenseJson();
  ASSERT_TRUE(defense.contains("hardEmbed"));
  EXPECT_TRUE(defense["hardEmbed"].value("sealed", false))
      << defense["hardEmbed"].value("detail", std::string());
  EXPECT_TRUE(defense["hardEmbed"].value("depImage", false));
  EXPECT_TRUE(defense["hardEmbed"].value("aslrImage", false));
}

TEST_F(HardEmbedTest, EstopBlocksInspectWithoutLooseningCleanText) {
  clearEnv("PHOENIX_SECURITY_ENABLED");
  SecurityObservatory::instance().resetForTests();
  phoenix::safety::EmergencyStop::instance().press("inspect");
  auto stopped = SecurityObservatory::instance().inspectText("ordinary notes");
  EXPECT_TRUE(stopped.blocked);
  EXPECT_EQ(stopped.reason, "emergency-stop");
  phoenix::safety::EmergencyStop::instance().resetForTesting();
  auto clean = SecurityObservatory::instance().inspectText("ordinary notes");
  EXPECT_FALSE(clean.blocked);
  EXPECT_FALSE(SecurityObservatory::instance().pluginEnabled());
}

#ifdef PHOENIX_SECURITY_LINE_MAIN
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
