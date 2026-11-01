#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "peer_hot_update.hpp"
#include "peer_evolve_extract.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <drogon/HttpClient.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace phoenix {
namespace peer {
namespace {

std::string envOr(const char *key, const std::string &fallback) {
  const char *v = std::getenv(key);
  if (!v || !*v) return fallback;
  return std::string(v);
}

bool forbiddenSource(const std::string &fn) {
  std::string low = fn;
  for (char &c : low) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  const char *needles[] = {
      "system(", "popen", "createprocess", "winexec", "ofstream",
      "fstream", "deletefile", "filesystem", "loadlibrary", "winapi",
      "../", "..\\"};
  for (const char *n : needles)
    if (low.find(n) != std::string::npos) return true;
  return false;
}

#ifdef _WIN32
struct ProcOut {
  int code{1};
  std::string text;
  bool timedOut{false};
};

std::string drainPipe(HANDLE h) {
  std::string s;
  DWORD avail = 0;
  char buf[4096];
  while (PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
    DWORD got = 0;
    const DWORD want = avail > sizeof(buf) ? static_cast<DWORD>(sizeof(buf)) : avail;
    if (!ReadFile(h, buf, want, &got, nullptr) || got == 0) break;
    s.append(buf, buf + got);
  }
  return s;
}

ProcOut runCaptured(const std::string &cmdline, const std::string &cwd,
                    DWORD timeoutMs) {
  ProcOut result;
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE outR = nullptr, outW = nullptr, errR = nullptr, errW = nullptr;
  if (!CreatePipe(&outR, &outW, &sa, 0) || !CreatePipe(&errR, &errW, &sa, 0)) {
    result.text = "pipe failed";
    return result;
  }
  SetHandleInformation(outR, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(errR, HANDLE_FLAG_INHERIT, 0);
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = outW;
  si.hStdError = errW;
  PROCESS_INFORMATION pi{};
  std::vector<char> cmd(cmdline.begin(), cmdline.end());
  cmd.push_back('\0');
  const BOOL started = CreateProcessA(
      nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
      cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
  CloseHandle(outW);
  CloseHandle(errW);
  if (!started) {
    CloseHandle(outR);
    CloseHandle(errR);
    result.text = "CreateProcess failed";
    return result;
  }
  const auto deadline = GetTickCount64() + timeoutMs;
  for (;;) {
    result.text += drainPipe(outR);
    result.text += drainPipe(errR);
    if (WaitForSingleObject(pi.hProcess, 50) == WAIT_OBJECT_0) break;
    if (GetTickCount64() > deadline) {
      TerminateProcess(pi.hProcess, 1);
      result.timedOut = true;
      break;
    }
  }
  result.text += drainPipe(outR);
  result.text += drainPipe(errR);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  result.code = result.timedOut ? 124 : static_cast<int>(code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  CloseHandle(outR);
  CloseHandle(errR);
  return result;
}
#endif

std::string usefulError(const std::string &text) {
  std::istringstream in(text);
  std::string line, last;
  while (std::getline(in, line)) {
    const auto p = line.find("error:");
    if (p != std::string::npos) last = line.substr(p, 220);
  }
  if (!last.empty()) return last;
  return text.size() > 220 ? text.substr(text.size() - 220) : text;
}

bool postAddon(const std::string &baseUrl, const std::string &token,
               const std::string &dllPath) {
  auto client = drogon::HttpClient::newHttpClient(baseUrl);
  if (!client) return false;
  auto req = drogon::HttpRequest::newHttpRequest();
  req->setMethod(drogon::Post);
  req->setPath("/api/addons/add");
  req->addHeader("Authorization", "Bearer " + token);
  req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  req->setBody(nlohmann::json{{"path", dllPath}}.dump());
  std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
  auto future = promise.get_future();
  client->sendRequest(req, [&promise](drogon::ReqResult result,
                                      const drogon::HttpResponsePtr &resp) {
    try {
      promise.set_value({result, resp});
    } catch (...) {
    }
  });
  if (future.wait_for(std::chrono::seconds(8)) != std::future_status::ready)
    return false;
  const auto got = future.get();
  if (got.first != drogon::ReqResult::Ok || !got.second) return false;
  return got.second->statusCode() >= 200 && got.second->statusCode() < 300;
}

}  // namespace

PeerPushResult pushEvolveToPeer(const std::string &goal, const std::string &deliverable) {
  const std::string peerUrl = envOr("AI_PEER_URL", "");
  if (peerUrl.empty()) return {};
  if (goal.find("PEER-EVOLVE") == std::string::npos) return {};
#ifndef _WIN32
  std::cout << "[peer-evolve] compile host is not Windows; skip" << std::endl;
  (void)deliverable;
  return {};
#else
  const std::string fn = extractEvolve(goal, deliverable);
  const bool selfMod = goal.find("SELF-MOD") != std::string::npos;
  const bool algoOpt =
      !selfMod && goal.find("ALGO-OPT") != std::string::npos;
  const bool scored = selfMod || algoOpt;
  if (fn.find("for") == std::string::npos && fn.find("while") == std::string::npos)
    return {};
  if (forbiddenSource(fn)) {
    std::cout << "[peer-evolve] reject blocked call" << std::endl;
    return {PeerPushResult::Kind::Rejected, "blocked call"};
  }
  static std::mutex mu;
  static std::unordered_map<std::string, std::string> pushed;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (pushed[peerUrl] == fn) return {};
  }

  const std::string selfName = envOr("AI_INSTANCE_ID", "alpha");
  const std::string peerName = envOr("AI_PEER_NAME", "beta");
  const std::string judge = envOr("PHOENIX_EVOLVE_JUDGE",
                                  "E:\\079\\v6.0Alixander\\phoenix_milestone_judge");
  const std::string gxx = envOr("PHOENIX_GXX", "C:\\msys64\\mingw64\\bin\\g++.exe");
  const std::string nlohmannInc = envOr(
      "PHOENIX_NLOHMANN",
      "E:\\079\\v6.0Alixander\\phoenix\\build\\conan_home\\p\\nlohm1bed1ddc0a2fa\\p\\include");
  const std::string repo = envOr("PHOENIX_REPO", "E:\\079\\v6.0Alixander\\phoenix");
  const auto stampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  const std::string evoId = "self_evolve_" + selfName + "_" + std::to_string(stampMs);
  const std::string peerDir = repo + "\\runtime_store\\peers\\" + peerName;
  const std::string buildDir = repo + "\\runtime_store\\peers\\" + selfName + "\\build";
  std::error_code ec;
  std::filesystem::create_directories(peerDir, ec);
  std::filesystem::create_directories(buildDir, ec);

  {
    std::ofstream hdr(buildDir + "\\evo_build.hpp", std::ios::binary);
    hdr << "#pragma once\n#define EVO_ID \"" << evoId << "\"\n";
    hdr << "#define EVO_STAMP \"runtime_store/peers/" << peerName << "/active.txt\"\n";
    hdr << "#define EVO_GOAL_TOKEN \"PEER-EVOLVE\"\n";
  }
  {
    std::ofstream src(buildDir + "\\user_fn.cpp", std::ios::binary);
    if (algoOpt)
      src << "#include <algorithm>\n#include <thread>\n#include <vector>\n";
    src << "#include <string>\n#include <cstdint>\nusing std::to_string;\n" << fn;
    if (fn.empty() || fn.back() != '\n') src << "\n";
  }
  const std::string harnessFile = selfMod ? "harness_self.cpp"
                                   : algoOpt ? "harness_opt.cpp"
                                             : "harness.cpp";
  const std::string optFlags = algoOpt ? " -pthread -fopenmp" : "";
  const ProcOut compiled = runCaptured(
      "\"" + gxx + "\" -std=c++17 -O2" + optFlags + " -I \"" + buildDir +
          "\" -o \"" + buildDir + "\\harness.exe\" \"" + judge + "\\" +
          harnessFile + "\"",
      buildDir, 40000);
  if (compiled.code != 0) {
    const std::string err = usefulError(compiled.text);
    std::cout << "[peer-evolve] harness compile " << err << std::endl;
    return {PeerPushResult::Kind::Rejected, err};
  }
  const ProcOut ran = runCaptured("\"" + buildDir + "\\harness.exe\"", buildDir,
                                 algoOpt ? 25000 : 8000);
  if (ran.code != 0 || ran.timedOut) {
    std::cout << "[peer-evolve] harness run code=" << ran.code << std::endl;
    return {PeerPushResult::Kind::Rejected,
            "run code=" + std::to_string(ran.code)};
  }
  std::string answer = ran.text;
  while (!answer.empty() &&
         (answer.back() == '\r' || answer.back() == '\n' || answer.back() == ' '))
    answer.pop_back();
  const auto nl = answer.find_last_of("\r\n");
  if (nl != std::string::npos) answer = answer.substr(nl + 1);
  if (answer.empty()) {
    std::cout << "[peer-evolve] harness empty" << std::endl;
    return {PeerPushResult::Kind::Rejected, "empty"};
  }
  if (scored && answer.rfind("OK ", 0) != 0) {
    std::cout << "[peer-evolve] scored " << answer.substr(0, 120) << std::endl;
    return {PeerPushResult::Kind::Rejected, answer.substr(0, 180)};
  }
  const std::string dll = peerDir + "\\" + evoId + ".dll";
  const ProcOut linked = runCaptured(
      "\"" + gxx + "\" -std=c++17 -O2" + optFlags + " -shared -I \"" + buildDir +
          "\" -I \"" + repo + "\" -I \"" + nlohmannInc + "\" -o \"" + dll +
          "\" \"" + judge + "\\wrapper.cpp\"",
      buildDir, 60000);
  if (linked.code != 0) {
    const std::string err = usefulError(linked.text);
    std::cout << "[peer-evolve] dll " << err << std::endl;
    return {PeerPushResult::Kind::Rejected, err};
  }
  {
    std::ofstream copy(peerDir + "\\" + evoId + ".cpp", std::ios::binary);
    copy << fn;
  }
  {
    std::ofstream stamp(peerDir + "\\active.txt", std::ios::binary);
    stamp << evoId << "\n";
  }
  if (!postAddon(peerUrl, envOr("AI_PEER_TOKEN", "local-fullstack"), dll)) {
    std::cout << "[peer-evolve] peer add failed " << peerUrl << std::endl;
    return {PeerPushResult::Kind::Idle, "peer add failed"};
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    pushed[peerUrl] = fn;
  }
  std::cout << "[peer-evolve] pushed " << evoId << " -> " << peerName
            << " harness=" << answer.substr(0, 32) << std::endl;
  return {PeerPushResult::Kind::Loaded, answer.substr(0, 64)};
#endif
}

std::string latestIncomingSource() {
  const std::string self = envOr("AI_INSTANCE_ID", "alpha");
  const std::string repo = envOr("PHOENIX_REPO", "E:\\079\\v6.0Alixander\\phoenix");
  std::ifstream stamp(repo + "\\runtime_store\\peers\\" + self + "\\active.txt");
  std::string id;
  if (!std::getline(stamp, id)) return {};
  while (!id.empty() &&
         (id.back() == '\r' || id.back() == '\n' || id.back() == ' '))
    id.pop_back();
  if (id.empty()) return {};
  std::ifstream in(repo + "\\runtime_store\\peers\\" + self + "\\" + id + ".cpp");
  if (!in) return {};
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace peer
}  // namespace phoenix
